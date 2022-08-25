#ifndef ALIGNMENTHELPER_H
#define ALIGNMENTHELPER_H
#include "./stacker.h"

class AlignmentHelper
{
	Stacker& _stacker;
	size_t _alignerIndex;
	std::mutex _mutex;

	AlignmentHelper(Stacker& stacker, size_t alignerIndex)
	: _stacker(stacker)
	, _alignerIndex(alignerIndex)
	{
		
	}

	void Job(uint32_t i)
	{
		_stacker._aligners[i]->Align(_stacker._stackingData[_alignerIndex].stars[i]);
		auto tileMatches = _stacker._aligners[i]->GetMatches();

		_mutex.lock();
		_stacker._matches.insert(tileMatches.begin(), tileMatches.end());
		_mutex.unlock();
	}

public:
	static void Run(Stacker& stacker, size_t alignerIndex)
	{
		AlignmentHelper helper(stacker, alignerIndex);
		auto [hTileCount, vTileCount] = GetTileCounts( stacker._width, stacker._height );
		oneapi::tbb::parallel_for( oneapi::tbb::blocked_range<int>( 0, hTileCount * vTileCount ), [&helper] ( const oneapi::tbb::blocked_range<int>& range )
		{
			for ( int i = range.begin(); i < range.end(); ++i )
			{
				helper.Job( i );
			}
		} );
	}
};

#endif
