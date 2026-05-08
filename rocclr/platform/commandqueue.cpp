/* Copyright (c) 2012 - 2021 Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#include "commandqueue.hpp"
#include "thread/monitor.hpp"
#include "device/device.hpp"
#include "platform/context.hpp"
#include <chrono>
#include <cstdlib>
#include <cstring>

/*!
 * \file commandQueue.cpp
 * \brief  Definitions for HostQueue object.
 *
 * \author Laurent Morichetti
 * \date   October 2008
 */

namespace amd {

HostQueue::HostQueue(Context& context, Device& device, cl_command_queue_properties props,
                     uint queueRTCUs, Priority priority, const std::vector<uint32_t>& cuMask)
    : CommandQueue(context, device, props, device.info().queueProperties_, queueRTCUs,
                   priority, cuMask),
      lastEnqueueCommand_(nullptr),
      head_(nullptr),
      tail_(nullptr),
      isActive_(false) {
  if (GPU_FORCE_QUEUE_PROFILING) {
    properties().set(CL_QUEUE_PROFILING_ENABLE);
  }
  if (AMD_DIRECT_DISPATCH) {
    // Initialize the queue
    thread_.Init(this);
  } else {
    if (thread_.state() >= Thread::INITIALIZED) {
      ScopedLock sl(queueLock_);
      thread_.start(this);
      queueLock_.wait();
    }
  }
}

bool HostQueue::terminate() {
  if (AMD_DIRECT_DISPATCH) {
    if (vdev() != nullptr) {
      // If the queue still has the last command, then wait and release it
      // We must be in protected way to get last command when calling
      // awaitCompletion() where lastCommand will be released and possibly
      // destroyed.
      Command* lastCommand = getLastQueuedCommand(true);
      if (lastCommand != nullptr) {
        // Check if CPU batch wasn't flushed for completion with the last command
        if (GetSubmissionBatch() != nullptr) {
            auto command = new Marker(*this, false);
            if (command != nullptr) {
              ClPrint(LOG_DEBUG, LOG_CMD, "Marker queued to ensure finish");
              command->enqueue();
              lastCommand = command;
            }
        }
        lastCommand->awaitCompletion();
        // Note that if lastCommand isn't a marker, it may not be lastEnqueueCommand_ now
        // after lastCommand->awaitCompletion() is called.
        if (lastEnqueueCommand_ != nullptr) {
          lastEnqueueCommand_ ->release(); // lastEnqueueCommand_ should be a marker
          lastEnqueueCommand_ = nullptr;
        }
        lastCommand->release();
      }
    }
    thread_.Release();
    thread_.acceptingCommands_ = false;
  } else {
    if (Os::isThreadAlive(thread_)) {
      Command* marker = nullptr;
      // Send a finish if the queue is still accepting commands.
      if (lastEnqueueCommand_ != nullptr || !amd::IS_HIP) {
        ScopedLock sl(queueLock_);
        if (thread_.acceptingCommands_) {
          marker = new Marker(*this, false);
          if (marker != nullptr) {
            append(*marker);
            queueLock_.notify();
          }
        }
      }
      // FIX-CQ1 V17.5-rc3: bound the non-direct-dispatch terminate() yield
      // loops. The original code had two unbounded yield loops:
      //   (1) wait for marker completion (marker->status() > CL_COMPLETE)
      //   (2) wait for thread finish (thread_.state() < Thread::FINISHED)
      // If a GPU fence is stuck, (1) spins forever. (2) has the same risk
      // after the FIXME_lmoriche comment. Both are now bounded by the same
      // service-survival deadline used by K-7.2 awaitCompletion.
      // On Linux AMD_DIRECT_DISPATCH=true by default so this path is not
      // normally taken, but defensive bounding prevents hang if someone
      // sets AMD_DIRECT_DISPATCH=0 or on non-Linux platforms.
      auto terminate_deadline = []{
        const char* hip_surv = std::getenv("HIP_SERVICE_SURVIVAL");
        const char* rocr_surv = std::getenv("ROCR_SERVICE_SURVIVAL");
        bool gated = (hip_surv && hip_surv[0] == '1') ||
                     (rocr_surv && rocr_surv[0] == '1');
        if (!gated) return std::chrono::steady_clock::time_point::max();
        uint64_t ms = 0;
        for (const char* name : {"HIP_AWAIT_FAIL_MS", "ROCR_SIGNAL_WAIT_MAX_MS"}) {
          const char* v = std::getenv(name);
          if (v && *v) { char* e; ms = std::strtoull(v, &e, 10); if (e != v && ms > 0) break; ms = 0; }
        }
        if (ms == 0) ms = 10000;
        return std::chrono::steady_clock::now() + std::chrono::milliseconds(ms);
      }();

      if (marker != nullptr) {
        if (marker->notifyCmdQueue()) {
          while (marker->status() > CL_COMPLETE && Os::isThreadAlive(thread_)) {
            if (std::chrono::steady_clock::now() > terminate_deadline) {
              ClPrint(LOG_WARNING, LOG_CMD,
                      "FIX-CQ1: terminate() marker wait timed out — breaking");
              break;
            }
            amd::Os::yield();
          }
        }
        marker->release();
      }

      // Wake-up the command loop, so it can exit
      {
        ScopedLock sl(queueLock_);
        thread_.acceptingCommands_ = false;
        queueLock_.notify();
      }

      // FIXME_lmoriche: fix termination handshake
      while (thread_.state() < Thread::FINISHED && Os::isThreadAlive(thread_)) {
        if (std::chrono::steady_clock::now() > terminate_deadline) {
          ClPrint(LOG_WARNING, LOG_CMD,
                  "FIX-CQ1: terminate() thread-finish wait timed out — breaking");
          break;
        }
        Os::yield();
      }
    }
  }

  if (Agent::shouldPostCommandQueueEvents()) {
    Agent::postCommandQueueFree(as_cl(this->asCommandQueue()));
  }

  return true;
}

void HostQueue::finish(bool cpu_wait) {
  // L2 V17.5-rc3-candidate: if the per-device AwaitDegraded latch is already
  // set (by an earlier finish() in this SyncAllStreams loop, or by any other
  // K-7.x path), skip the IsHwEventReady (~HIP_AWAIT_FAIL_MS) wait below.
  // Cures the N x HIP_AWAIT_FAIL_MS compounding seen in device_sync_hang
  // (128 streams x 2 s = 256 s without this check) by short-circuiting every
  // stream after the first that triggered the latch.
  if (vdev() != nullptr && vdev()->device().AwaitDegraded()) {
    Command* cmd_ld = getLastQueuedCommand(true);
    if (cmd_ld != nullptr) {
      cmd_ld->setStatus(CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST);
      ScopedLock sl(vdev()->execution());
      ScopedLock l(lastCmdLock_);
      if (cmd_ld == lastEnqueueCommand_) {
        lastEnqueueCommand_->release();
        lastEnqueueCommand_ = nullptr;
      }
      cmd_ld->release();
    }
    return;
  }

  Command* command = nullptr;
  if (IS_HIP) {
    command = getLastQueuedCommand(true);
    if (command == nullptr) {
      assert(GetSubmissionBatch() == nullptr &&
        "Can't claim the queue is finished with the active batch!");
      return;
    }
  }
  // Force marker if the batch wasn't sent for CPU update or fence is dirty
  if (nullptr == command || (GetSubmissionBatch() != nullptr) || vdev()->isFenceDirty()) {
    if (nullptr != command) {
      command->release();
    }
    // Send a finish to make sure we finished all commands
    command = new Marker(*this, false);
    if (command == NULL) {
      return;
    }
    ClPrint(LOG_DEBUG, LOG_CMD, "Marker queued to ensure finish");
    command->enqueue();
  }
  // Check HW status of the ROCcrl event. Note: not all ROCclr modes support HW status
  static constexpr bool kWaitCompletion = true;
  const bool hwReady = device().IsHwEventReady(command->event(), kWaitCompletion);
  if (!hwReady) {
    // V17.4 service-survival early-return: if the underlying ROCr/HSA HW
    // event already returned a not-ready (the ROCr signal-wait gave up at
    // its own deadline -- e.g. ROCR_SIGNAL_WAIT_MAX_MS), short-circuit the
    // CLR-layer awaitCompletion below and mark the command as errored.
    //
    // K-7.8 V17.5 update: also accept the new HIP_SERVICE_SURVIVAL gate
    // (K-7.6 split CLR/ROCr env names; this path was previously gated only
    // on the ROCr-layer name, which forced customers to keep both envs set
    // even if they only meant to enable CLR-layer survival), and crucially
    // also raise the K-7.4 per-device awaitCompletion-degraded latch so
    // that K-7.7 + K-7.8's API-level exit checks (hipStreamSynchronize,
    // hipDeviceSynchronize, hipEventSynchronize, ihipMemcpy, ihipMemset,
    // ...) propagate hipErrorNotReady to the customer service. Without
    // setting the latch, this V17.4 path silently returned hipSuccess to
    // the application -- defeating the whole point of the 4 s reject
    // contract: the app would believe its kernel/copy succeeded and
    // happily consume a stale buffer.
    const char* rocr_survival = std::getenv("ROCR_SERVICE_SURVIVAL");
    const char* hip_survival = std::getenv("HIP_SERVICE_SURVIVAL");
    const bool gated_on =
        (rocr_survival != nullptr && std::strcmp(rocr_survival, "1") == 0) ||
        (hip_survival  != nullptr && std::strcmp(hip_survival,  "1") == 0);
    if (gated_on) {
      ClPrint(LOG_ERROR, LOG_CMD,
              "SERVICE_SURVIVAL: HW event wait failed; marking command as error "
              "and raising per-device degraded latch");
      command->setStatus(CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST);
      device().SetAwaitDegraded();
      return;
    }
  }
  if (cpu_wait || !hwReady) {
    ClPrint(LOG_DEBUG, LOG_CMD, "HW Event not ready, awaiting completion instead");
    command->awaitCompletion();

    if (IS_HIP) {
      ScopedLock sl(vdev()->execution());
      ScopedLock l(lastCmdLock_);
      // Runtime can clear the last command only if no other submissions occured
      // during finish()
      if (command == lastEnqueueCommand_) {
        lastEnqueueCommand_->release();
        lastEnqueueCommand_ = nullptr;
      }
    }
  }

  command->release();
  ClPrint(LOG_DEBUG, LOG_CMD, "All commands finished");
}

void HostQueue::loop(device::VirtualDevice* virtualDevice) {
  // Notify the caller that the queue is ready to accept commands.
  {
    ScopedLock sl(queueLock_);
    thread_.acceptingCommands_ = true;
    queueLock_.notify();
  }
  // Create a command batch with all the commands present in the queue.
  Command* head = NULL;
  Command* tail = NULL;
  while (true) {
    // Get one command from the queue
    Command* command = queue_.dequeue();
    if (command == NULL) {
      ScopedLock sl(queueLock_);
      while ((command = queue_.dequeue()) == NULL) {
        if (!thread_.acceptingCommands_) {
          return;
        }
        queueLock_.wait();
      }
    }

    command->retain();

    // Process the command's event wait list.
    const Command::EventWaitList& events = command->eventWaitList();
    bool dependencyFailed = false;
    ClPrint(LOG_DEBUG, LOG_CMD, "Command (%s) processing: %p ,events.size(): %d",
            amd::activity_prof::getOclCommandKindString(command->type()), command, events.size());
    for (const auto& it : events) {
      // Only wait if the command is enqueued into another queue.
      if (it->command().queue() != this) {
        // Runtime has to flush the current batch only if the dependent wait is blocking
        if (it->command().status() != CL_COMPLETE) {
          ClPrint(LOG_DEBUG, LOG_CMD, "Command (%s) %p awaiting event: %p",
                  amd::activity_prof::getOclCommandKindString(command->type()),
                  command, it);
          virtualDevice->flush(head, true);
          tail = head = NULL;
          dependencyFailed |= !it->awaitCompletion();
        }
      }
    }

    // Insert the command to the linked list.
    if (NULL == head) {  // if the list is empty
      head = tail = command;
    } else {
      tail->setNext(command);
      tail = command;
    }

    if (dependencyFailed) {
      command->setStatus(CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST);
      continue;
    }

    ClPrint(LOG_DEBUG, LOG_CMD, "Command (%s) submitted: %p",
            amd::activity_prof::getOclCommandKindString(command->type()),
            command);

    command->setStatus(CL_SUBMITTED);

    // Submit to the device queue.
    command->submit(*virtualDevice);

    // if this is a user invisible marker command, then flush
    if (0 == command->type()) {
      virtualDevice->flush(head);
      tail = head = NULL;
    }
  }  // while (true) {
}

void HostQueue::append(Command& command) {
  // We retain the command here. It will be released when its status
  // changes to CL_COMPLETE
  if ((command.getWaitBits() & 0x1) != 0) {
    finish();
  }
  command.retain();
  command.setStatus(CL_QUEUED);
  // V17.5 firewall: bump the per-stream pending counter symmetrically
  // with the retain above. Decrement happens in Event::setStatus when
  // the command transitions to CL_COMPLETE (see command.cpp).
  IncPending();
  queue_.enqueue(&command);
  if (!IS_HIP) {
    return;
  }

  // Set last submitted command
  Command* prevLastEnqueueCommand = nullptr;

  // Attach only real commands and skip internal notifications for CPU queue
  if (command.waitingEvent() == nullptr) {
    command.retain();

    // lastCmdLock_ ensures that lastEnqueueCommand() can retain the command before it is swapped
    // out. We want to keep this critical section as short as possible, so the command should be
    // released outside this section.
    ScopedLock l(lastCmdLock_);

    prevLastEnqueueCommand = lastEnqueueCommand_;
    lastEnqueueCommand_ = &command;
  }

  if (prevLastEnqueueCommand != nullptr) {
    prevLastEnqueueCommand->release();
  }
}

bool HostQueue::isEmpty() {
  // Get a snapshot of queue size
  return queue_.empty();
}

Command* HostQueue::getLastQueuedCommand(bool retain) {
  if (AMD_DIRECT_DISPATCH) {
    // The batch update must be lock protected to avoid a race condition
    // when multiple threads submit/flush/update the batch at the same time
    ScopedLock sl(vdev()->execution());
    // Since the lastCmdLock_ is acquired, it is safe to read and retain the lastEnqueueCommand.
    // It is guaranteed that the pointer will not change.
    if (retain && lastEnqueueCommand_ != nullptr) {
      lastEnqueueCommand_->retain();
    }
    return lastEnqueueCommand_;
  } else {
    // Get last submitted command
    ScopedLock l(lastCmdLock_);

    // Since the lastCmdLock_ is acquired, it is safe to read and retain the lastEnqueueCommand.
    // It is guaranteed that the pointer will not change.
    if (retain && lastEnqueueCommand_ != nullptr) {
      lastEnqueueCommand_->retain();
    }
    return lastEnqueueCommand_;
  }
}

DeviceQueue::~DeviceQueue() {
  delete virtualDevice_;
  ScopedLock lock(context().lock());
  context().removeDeviceQueue(device(), this);
}

bool DeviceQueue::create() {
  const bool defaultDeviceQueue = properties().test(CL_QUEUE_ON_DEVICE_DEFAULT);
  bool result = false;

  virtualDevice_ = device().createVirtualDevice(this);
  if (virtualDevice_ != NULL) {
    result = true;
    context().addDeviceQueue(device(), this, defaultDeviceQueue);
  }

  return result;
}

}  // namespace amd
