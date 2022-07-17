#ifndef IPARALLEL_H
#define IPARALLEL_H

#include <thread>
#include <mutex>
#include <vector>

class IParallel
{
protected:
    uint32_t _threadCount;
    uint32_t _jobCount;

    std::vector<std::thread> _threads;
    std::mutex _mutex;

    IParallel( uint32_t jobCount = 1 )
        : _threadCount( std::min( jobCount, std::thread::hardware_concurrency() ) )
        , _jobCount( jobCount )
    {

    }

    virtual void Job( uint32_t index ) = 0;

    void DoParallelJobs()
    {
        const auto div = _jobCount / _threadCount;
        const auto mod = _jobCount % _threadCount;

        for ( uint32_t i = 0; i < _threadCount; ++i )
        {
            const auto rangeStart = i * div + std::min( i, mod );
            const auto rangeSize = div + ( ( i < mod ) ? 1 : 0 );
            _threads.emplace_back( [this, rangeSize, rangeStart] ()
            {
                for ( auto n = rangeStart; n < rangeStart + rangeSize; ++n )
                {
                    this->Job( n );
                }
            } );
        }

        for ( uint32_t i = 0; i < _threadCount; ++i )
        {
            _threads[i].join();
        }

        _threads.clear();
    }

    void SetJobCount( uint32_t jobCount )
    {
        _jobCount = jobCount;
        _threadCount = std::min( jobCount, std::thread::hardware_concurrency() );
    }
};

#endif
