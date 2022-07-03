#ifndef ALIGNMENTHELPER_H
#define ALIGNMENTHELPER_H
#include "../Core/IParallel.h"
#include "./stacker.h"

class AlignmentHelper : public IParallel
{
	Stacker& _stacker;
	size_t _alignerIndex;

	AlignmentHelper(Stacker& stacker, size_t alignerIndex)
	: IParallel(stacker._hTileCount* stacker._vTileCount)
	, _stacker(stacker)
	, _alignerIndex(alignerIndex)
	{
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
