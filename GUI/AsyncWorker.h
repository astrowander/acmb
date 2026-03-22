#pragma once
#include "./../Core/macros.h"

#include <atomic>
#include <functional>
#include <future>
#include <mutex>
#include <string>

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

    using TaskFunc = std::function<void(std::function<bool(float)>)>;

    AsyncWorker() = default;
    ~AsyncWorker()
    {
        Cancel();
        Wait();
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
        _status.store(Status::Running);
        _progress.store(0.0f);
        _cancelRequested.store(false);
        {
            std::lock_guard lock(_errorMutex);
            _errorMessage.clear();
        }

        _future = std::async(std::launch::async, [this, task = std::move(task)]()
        {
            try
            {
                task([this](float progress) -> bool
                {
                    _progress.store(std::clamp(progress, 0.0f, 1.0f));
                    return !_cancelRequested.load();
                });

                if ( _cancelRequested.load() )
                    _status.store(Status::Cancelled);
                else
                    _status.store(Status::Completed);
            }
            catch ( const std::exception& e )
            {
                std::lock_guard lock(_errorMutex);
                _errorMessage = e.what();
                _status.store(Status::Failed);
            }
            catch ( ... )
            {
                std::lock_guard lock(_errorMutex);
                _errorMessage = "Unknown error";
                _status.store(Status::Failed);
            }
        }).share();
    }

    /// Requests cancellation of the running task.
    void Cancel()
    {
        _cancelRequested.store(true);
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
        return _progress.load();
    }

    /// Returns the current status.
    Status GetStatus() const
    {
        return _status.load();
    }

    /// Returns true if the worker is currently executing a task.
    bool IsRunning() const
    {
        return _status.load() == Status::Running;
    }

    /// Returns true if cancellation was requested.
    bool IsCancellationRequested() const
    {
        return _cancelRequested.load();
    }

    /// Returns the error message if the task failed.
    std::string GetErrorMessage() const
    {
        std::lock_guard lock(_errorMutex);
        return _errorMessage;
    }

    /// Resets the worker back to Idle state. Must not be called while running.
    void Reset()
    {
        if ( _status.load() == Status::Running )
            return;

        Wait();
        _status.store(Status::Idle);
        _progress.store(0.0f);
        _cancelRequested.store(false);
        {
            std::lock_guard lock(_errorMutex);
            _errorMessage.clear();
        }
    }

private:

    std::shared_future<void> _future;
    std::atomic<Status> _status{ Status::Idle };
    std::atomic<float> _progress{ 0.0f };
    std::atomic<bool> _cancelRequested{ false };

    mutable std::mutex _errorMutex;
    std::string _errorMessage;
};

ACMB_GUI_NAMESPACE_END
