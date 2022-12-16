
#include "GatherElementsPlugin.h"

#include <assert.h>

#include <chrono>
#include <stdio.h>
//#include "amirCommon.h"
//#include "cuda_utils.h"
//#include "common.h"
#include "serialize.hpp"
#include "torch_gather.h"
#include <string.h>
#include <cuda_runtime.h>
//namespace amirstan {
namespace plugin {

namespace {
static const char *PLUGIN_VERSION{"1"};
static const char *PLUGIN_NAME{"GatherElements"};
}  // namespace


nvinfer1::PluginFieldCollection GatherElementsPluginDynamicCreator::mFC{};
std::vector<nvinfer1::PluginField> GatherElementsPluginDynamicCreator::mPluginAttributes({});


GatherElementsPluginDynamic::GatherElementsPluginDynamic(const std::string &name, int dim):
    mLayerName(name), mDim(dim)
    {

    }
    // ,mnpoint(npoint){
    //   cudaMallocManaged(&temp,npoint*sizeof(float));
       
    //   cudaMemset(temp,float(1e10),npoint*sizeof(float));
       
    //    }

GatherElementsPluginDynamic::GatherElementsPluginDynamic(const std::string name,const void *data,size_t length):
    mLayerName(name){
      //deserialize_value(&data, &length);
      // cudaMallocManaged(&temp,mnpoint*sizeof(float));

    }

GatherElementsPluginDynamic::~GatherElementsPluginDynamic(){
  // cudaFree(temp);
}

nvinfer1::IPluginV2DynamicExt *GatherElementsPluginDynamic::clone() const noexcept{
  GatherElementsPluginDynamic *plugin = new GatherElementsPluginDynamic(mLayerName, mDim);
  plugin->setPluginNamespace(getPluginNamespace());

  return plugin;
}


nvinfer1::DimsExprs GatherElementsPluginDynamic::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs,
    nvinfer1::IExprBuilder &exprBuilder) noexcept{
  return inputs[1];
}

bool GatherElementsPluginDynamic::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc *inOut, int nbInputs,
    int nbOutputs) noexcept{
  assert(0 <= pos && pos < 3);
  const auto *in = inOut;
  const auto *out = inOut + nbInputs;

  // switch (pos) {
  //   case 0:
  //     return (in[0].type == nvinfer1::DataType::kFLOAT &&
  //             in[0].format == nvinfer1::TensorFormat::kLINEAR) ||
  //            //  (in[0].type == nvinfer1::DataType::kHALF &&
  //            //   in[0].format == nvinfer1::TensorFormat::kLINEAR) ||
  //            (in[0].type == nvinfer1::DataType::kINT32 &&
  //             in[0].format == nvinfer1::TensorFormat::kLINEAR);
  //   case 1:
  //     return in[1].type == nvinfer1::DataType::kINT32 &&
  //            in[1].format == nvinfer1::TensorFormat::kLINEAR;
  //   case 2:
  //     return out[0].type == in[0].type && out[0].format == in[0].format;
  // }
  switch (pos) {
    case 0:
      return (in[0].type == nvinfer1::DataType::kFLOAT &&
              in[0].format == nvinfer1::TensorFormat::kLINEAR);
    case 1:
      return in[1].type == in[0].type && in[1].format == in[0].format;
    case 2:
      return out[0].type == in[0].type && out[0].format == in[0].format;
  }
  return false;
}

void GatherElementsPluginDynamic::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc *inputs, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc *outputs,
    int nbOutputs) noexcept{
  // Validate input arguments
  assert(nbOutputs == 1);
  assert(nbInputs == 2);
}

size_t GatherElementsPluginDynamic::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc *inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc *outputs,
    int nbOutputs) const noexcept{
  return 0;
}

int GatherElementsPluginDynamic::enqueue(
    const nvinfer1::PluginTensorDesc *inputDesc,
    const nvinfer1::PluginTensorDesc *outputDesc, const void *const *inputs,
    void *const *outputs, void *workSpace,
    cudaStream_t stream) noexcept{
  nvinfer1::Dims input_dims = inputDesc[0].dims;
  nvinfer1::Dims index_dims = inputDesc[1].dims;

  auto data_type = inputDesc[0].type;

  switch (data_type) {
    case nvinfer1::DataType::kFLOAT:
      torch_gather<float>((float *)outputs[0], (float *)inputs[0],
                          (int *)inputs[1], mDim, &(input_dims.d[0]),
                          &(index_dims.d[0]), input_dims.nbDims, stream);
      break;

    case nvinfer1::DataType::kHALF:
      torch_gather<half>((half *)outputs[0], (half *)inputs[0],
                         (int *)inputs[1], mDim, &(input_dims.d[0]),
                         &(index_dims.d[0]), input_dims.nbDims, stream);
      break;

    case nvinfer1::DataType::kINT32:
      torch_gather<int>((int *)outputs[0], (int *)inputs[0], (int *)inputs[1],
                        mDim, &(input_dims.d[0]), &(index_dims.d[0]),
                        input_dims.nbDims, stream);
      break;
    default:
      return 1;
      break;
  }

  return 0;
}

nvinfer1::DataType GatherElementsPluginDynamic::getOutputDataType(
    int index, const nvinfer1::DataType *inputTypes,
    int nbInputs) const noexcept{
  assert(nbInputs == 2);
  return inputTypes[0];
}

// IPluginV2 Methods
const char *GatherElementsPluginDynamic::getPluginType() const noexcept{
  return PLUGIN_NAME;
}

const char *GatherElementsPluginDynamic::getPluginVersion() const noexcept{
  return PLUGIN_VERSION;
}

int GatherElementsPluginDynamic::getNbOutputs() const noexcept{ return 1; }

size_t GatherElementsPluginDynamic::getSerializationSize() const noexcept{
  return sizeof(mDim);
}

int GatherElementsPluginDynamic::initialize() noexcept{ return 0; }

void GatherElementsPluginDynamic::terminate() noexcept{}

// size_t GatherElementsPluginDynamic::getSerializationSize() const noexcept{
  
// }

void GatherElementsPluginDynamic::serialize(void *buffer) const noexcept{
  serialize_value(&buffer, mDim);
}

void GatherElementsPluginDynamic::destroy() noexcept{
  // This gets called when the network containing plugin is destroyed
  delete this;
}

void GatherElementsPluginDynamic::setPluginNamespace(const char *libNamespace) noexcept{
  mNamespace = libNamespace;
}

const char *GatherElementsPluginDynamic::getPluginNamespace() const noexcept{
  return mNamespace.c_str();
}
////////////////////// creator /////////////////////////////

GatherElementsPluginDynamicCreator::GatherElementsPluginDynamicCreator() {
  //mPluginAttributes = std::vector<PluginField>({PluginField("repeat_dims")});
  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char *GatherElementsPluginDynamicCreator::getPluginName() const
    noexcept{
  return PLUGIN_NAME;
}

const char *GatherElementsPluginDynamicCreator::getPluginVersion() const
    noexcept{
  return PLUGIN_VERSION;
}

const nvinfer1::PluginFieldCollection *GatherElementsPluginDynamicCreator::getFieldNames() noexcept{
  return &mFC;
}

nvinfer1::IPluginV2 *GatherElementsPluginDynamicCreator::createPlugin(
    const char *name, const nvinfer1::PluginFieldCollection *fc) noexcept{
  int dim = 0;

  for (int i = 0; i < fc->nbFields; i++) {
    if (fc->fields[i].data == nullptr) {
      continue;
    }
    std::string field_name(fc->fields[i].name);

    if (field_name.compare("dim") == 0) {
      dim = static_cast<const int *>(fc->fields[i].data)[0];
    }
  }

  GatherElementsPluginDynamic *plugin = new GatherElementsPluginDynamic(name, dim);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

nvinfer1::IPluginV2 *GatherElementsPluginDynamicCreator::deserializePlugin(
    const char *name, const void *serialData,
    size_t serialLength) noexcept{
  // This object will be deleted when the network is destroyed, which will
  // call FCPluginDynamic::destroy()
  auto plugin = new GatherElementsPluginDynamic(name, serialData, serialLength);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

void GatherElementsPluginDynamicCreator::setPluginNamespace(
    const char *libNamespace) noexcept{
  mNamespace = libNamespace;
}

const char *GatherElementsPluginDynamicCreator::getPluginNamespace() const noexcept{
  return mNamespace.c_str();
}

REGISTER_TENSORRT_PLUGIN(GatherElementsPluginDynamicCreator);

}  // namespace plugin
//}  // namespace amirstan
