#pragma once
#include "basetransform.h"
#include <array>
#include <functional>

ACMB_NAMESPACE_BEGIN
/// <summary>
/// Changes values of pixel channels according to given transformations
/// </summary>
class ChannelEqualizer : public BaseTransform
{
protected:
	ChannelEqualizer(IBitmapPtr pSrcBitmap);

public:
	/// Creates instance with source bitmap and given channel transformations. Number of transforms must be equal to the numuber of channels
	static std::shared_ptr<ChannelEqualizer> Create(IBitmapPtr pSrcBitmap, const std::vector< std::function<uint32_t( uint32_t )>>& channelTransforms );
	/// returns result with default transformations
	static IBitmapPtr AutoEqualize( IBitmapPtr pSrcBitmap );
};

/// <summary>
/// This class is needed for compatibility with pipelines. It applies default transformations
/// </summary>
class AutoChannelEqualizer : public BaseTransform
{
public:
	struct Settings {};
	/// Creates instance with source bitmap (null by default)
	AutoChannelEqualizer( IBitmapPtr pSrcBitmap = nullptr );
	/// Runs the transformation
	virtual void Run() override;
	/// Creates instance with given pixel format. Source bitmap must be set later
	static std::shared_ptr<AutoChannelEqualizer> Create( PixelFormat, Settings = {} );

	virtual void ValidateSettings() override;
};

ACMB_NAMESPACE_END