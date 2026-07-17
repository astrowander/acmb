#pragma once
#include "./../Core/macros.h"

#include <atomic>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

ACMB_GUI_NAMESPACE_BEGIN

/// Thread-safe asynchronous worker that can run any task, report progress, and be cancelled.
class AsyncWorker
{
public:
    enum class Status
    {
        Idle,
        Running,
        Completed,
        Cancelled,
        Failed
    };

    using ProgressFunc = std::function<bool(float)>;
    using TaskFunc = std::function<std::string(ProgressFunc)>;


    AsyncWorker() = default;
    ~AsyncWorker()
    {
        if ( _state )
            _state->cancelRequested.store(true);

        // Move the future to a short-lived background thread so the calling
        // (main) thread is never blocked waiting for the task to finish.
        // The shared State keeps all task-internal data alive until it completes.
        if ( _future.valid() )
            std::thread( [f = std::move( _future )] {} ).detach();
    }

    AsyncWorker(const AsyncWorker&) = delete;
    AsyncWorker& operator=(const AsyncWorker&) = delete;

    AsyncWorker(AsyncWorker&&) = default;
    AsyncWorker& operator=(AsyncWorker&&) = default;

    /// Starts a task asynchronously. The task receives a progress callback
    /// that returns false when cancellation is requested.
    /// Usage:
    ///   worker.Start([](auto reportProgress) {
    ///       for (int i = 0; i < 100; ++i) {
    ///           if (!reportProgress(i / 100.0f))
    ///               return; // cancelled
    ///           // ... do work ...
    ///       }
    ///   });
    void Start(TaskFunc task)
    {
        // Create fresh shared state so the running lambda does not capture
        // a raw 'this' pointer – the AsyncWorker can be moved or destroyed
        // without becoming a dangling pointer inside the async task.
        _state = std::make_shared<State>();
        _state->status.store(Status::Running);

        auto state = _state;
        _future = std::async(std::launch::async, [state, task = std::move(task)]()
        {
            try
            {
                task([state](float progress) -> bool
                {
                    state->progress.store(std::clamp(progress, 0.0f, 1.0f));
                    return !state->cancelRequested.load();
                });

                if ( state->cancelRequested.load() )
                    state->status.store(Status::Cancelled);
                else
                    state->status.store(Status::Completed);
            }
            catch ( const std::exception& e )
            {
                std::lock_guard lock(state->errorMutex);
                state->errorMessage = e.what();
                state->status.store(Status::Failed);
            }
            catch ( ... )
            {
                std::lock_guard lock(state->errorMutex);
                state->errorMessage = "Unknown error";
                state->status.store(Status::Failed);
            }
        }).share();
    }

    /// Requests cancellation of the running task.
    void Cancel()
    {
        if ( _state )
            _state->cancelRequested.store(true);
    }

    /// Blocks until the task finishes (or does nothing if idle).
    void Wait()
    {
        if ( _future.valid() )
            _future.get();
    }

    /// Returns the current progress in [0.0, 1.0].
    float GetProgress() const
    {
        return _state ? _state->progress.load() : 0.0f;
    }

    /// Returns the current status.
    Status GetStatus() const
    {
        return _state ? _state->status.load() : Status::Idle;
    }

    /// Returns true if the worker is currently executing a task.
    bool IsRunning() const
    {
        return GetStatus() == Status::Running;
    }

    /// Returns true if cancellation was requested.
    bool IsCancellationRequested() const
    {
        return _state ? _state->cancelRequested.load() : false;
    }

    /// Returns the error message if the task failed.
    std::string GetErrorMessage() const
    {
        if ( !_state )
            return {};
        std::lock_guard lock(_state->errorMutex);
        return _state->errorMessage;
    }

    /// Resets the worker back to Idle state. Must not be called while running.
    void Reset()
    {
        if ( GetStatus() == Status::Running )
            return;

        Wait();
        if ( _state )
        {
            _state->status.store(Status::Idle);
            _state->progress.store(0.0f);
            _state->cancelRequested.store(false);
            std::lock_guard lock(_state->errorMutex);
            _state->errorMessage.clear();
        }
    }

private:

    struct State
    {
        std::atomic<Status> status{ Status::Idle };
        std::atomic<float> progress{ 0.0f };
        std::atomic<bool> cancelRequested{ false };
        mutable std::mutex errorMutex;
        std::string errorMessage;
    };

    std::shared_ptr<State> _state = std::make_shared<State>();
    std::shared_future<void> _future;
};

ACMB_GUI_NAMESPACE_END
