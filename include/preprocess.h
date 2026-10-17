// Image preprocessing for TRELLIS.2: load -> white-bg removal -> square crop ->
// premultiply on black -> resize -> ImageNet normalize -> [3,S,S] torch CHW.
#pragma once
#include <vector>
#include <string>

namespace trellis {
struct Model;
// Returns [3*S*S] float in torch CHW order (== ggml [S,S,3,1]), or empty on failure.
std::vector<float> preprocess_image(const std::string& path, int S = 512);

// Non-neural background removal: existing alpha if present, else white-bg threshold;
// bbox-crop + premultiply on transparent black -> square RGBA uint8 cutout (size in `sz`).
std::vector<unsigned char> threshold_cutout(const std::string& path, int& sz);

// Width and height from the file header, without decoding. False when the file cannot be read.
bool image_size(const std::string& path, int& W, int& H);

// True when the image carries a real (not all-opaque) alpha channel, i.e. it is
// pre-matted and needs no background removal.
bool image_has_alpha(const std::string& path);

// BiRefNet background removal: run the matte, restore it to the source aspect ratio, then
// bbox-crop and premultiply -> square RGBA uint8 cutout
// (size returned in `sz`). bm = loaded birefnet model, gpu = device for the deform kernel.
std::vector<unsigned char> birefnet_cutout(const std::string& path, const Model& bm, int gpu, int& sz);
// "Views as framed" (multi-view): the same mattes, but the picture is returned exactly as given
// (premultiplied RGBA, same size, no bounding-box crop, no rescale, no recentring), because the
// camera of that view describes its framing. Square pictures only. An existing alpha is used as
// is, exactly as upstream inference_mv.py does.
std::vector<unsigned char> threshold_framed(const std::string& path, int& sz);
std::vector<unsigned char> birefnet_framed(const std::string& path, const Model& bm, int gpu, int& sz);

// Resize a square RGB/RGBA-uint8 cutout to SxS, ImageNet-normalize -> [3,S,S] torch CHW.
std::vector<float> normalize_cutout(const std::vector<unsigned char>& rgb, int sz, int S);

// Resize a square RGB-uint8 cutout to SxS and scale to [0,1] WITHOUT ImageNet normalization
// -> [3,S,S] torch CHW. The NAF guide branch (Pixal3D) consumes the raw [0,1] image, not the
// normalized one that feeds DINOv3.
std::vector<float> cutout_to_chw01(const std::vector<unsigned char>& rgb, int sz, int S);
} // namespace trellis
