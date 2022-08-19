#ifndef ALIGNMENTHELPER_H
#define ALIGNMENTHELPER_H
#include "../Core/IParallel.h"
#include "./stacker.h"

class AlignmentHelper final: public IParallel
{
	Stacker& _stacker;
	size_t _alignerIndex;

	AlignmentHelper(Stacker& stacker, size_t alignerIndex)
	: _stacker(stacker)
	, _alignerIndex(alignerIndex)
	{
		auto [hTileCount, vTileCount] = GetTileCounts( _stacker._width, _stacker._height );
		SetJobCount( hTileCount * vTileCount );
	}

	void Job(uint32_t i) override
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
		/*for (uint32_t i = 0; i < helper._jobCount; ++i)
		{
			helper.Job(i);
		}*/

		helper.DoParallelJobs();
	}
};

#endif
