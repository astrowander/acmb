#ifndef IPARALLEL_H
#define IPARALLEL_H

#include <thread>
#include <mutex>
#include <vector>

class IParallel
{
protected:
	const uint32_t _threadCount;
	
	std::vector<std::thread> _threads;
	std::mutex _mutex;

	IParallel(uint32_t jobCount)
	: _threadCount(std::min(jobCount, std::thread::hardware_concurrency()))
	{

	}

	virtual void Job(uint32_t index) = 0;

	void DoParallelJobs()
	{		
		for (uint32_t i = 0; i < _threadCount; ++i)
		{
			_threads.emplace_back(&IParallel::Job, this, i);
		}

		for (uint32_t i = 0; i < _threadCount; ++i)
		{
			_threads[i].join();
		}

		_threads.clear();
	}
};

#endif
