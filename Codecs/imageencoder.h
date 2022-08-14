#ifndef IMAGEENCODER_H
#define IMAGEENCODER_H

#include <string>
#include <ostream>
#include <memory>
#include <unordered_set>

class IBitmap;

class ImageEncoder
{
protected:
    std::shared_ptr<std::ostream> _pStream;
    inline static std::unordered_set<std::string> _allExtensions;

public:

    virtual void Attach(std::shared_ptr<std::ostream> pStream);
    virtual void Attach(const std::string& fileName);
    virtual void Detach();

    virtual ~ImageEncoder() = default;

    virtual void WriteBitmap(std::shared_ptr<IBitmap> pBitmap) = 0;

    static std::shared_ptr<ImageEncoder> Create(const std::string& fileName);

    static const std::unordered_set<std::string>& GetAllExtensions()
    {
        return _allExtensions;
    }
protected:
    static bool AddCommonExtensions( const std::unordered_set<std::string>& extensions )
    {
        _allExtensions.insert( std::begin( extensions ), std::end( extensions ) );
        return true;
    }
};

#define ADD_EXTENSIONS inline static bool handle = AddCommonExtensions(GetExtensions());
#endif // IMAGEENCODER_H
