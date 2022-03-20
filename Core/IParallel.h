#ifndef IPARALLEL_H
#define IPARALLEL_H

#include <thread>
#include <mutex>
#include <vector>

class IParallel
{
protected:
	const uint32_t _threadCount;
	const uint32_t _jobCount;

	std::vector<std::thread> _threads;
	std::mutex _mutex;

	IParallel(uint32_t jobCount)
	: _threadCount(std::min(jobCount, std::thread::hardware_concurrency()))
	, _jobCount(jobCount)
	{

	}

	virtual void Job(uint32_t index) = 0;

	void DoParallelJobs()
	{		
		const auto div = _jobCount / _threadCount;
		const auto mod = _jobCount % _threadCount;

		for (uint32_t i = 0; i < _threadCount; ++i)
		{
			const auto rangeStart = i * div + std::min(i, mod);
			const auto rangeSize = div + ((i < mod) ? 1 : 0);
			_threads.emplace_back([this, rangeSize, rangeStart]() { for (size_t n = rangeStart; n < rangeStart + rangeSize; ++n) { this->Job(n); }});
		}		

		for (uint32_t i = 0; i < _threadCount; ++i)
		{
			_threads[i].join();
		}

		_threads.clear();
	}
};

#endif
