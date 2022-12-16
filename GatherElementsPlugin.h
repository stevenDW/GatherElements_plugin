#pragma once

#include <cublas_v2.h>

#include <memory>
#include <string>
#include <vector>

//#include "../common/layer_base.h"

#include "NvInferPlugin.h"
//namespace amirstan {
namespace plugin {

class GatherElementsPluginDynamic : public nvinfer1::IPluginV2DynamicExt {
 public:
  GatherElementsPluginDynamic(const std::string &name, int dim);

  GatherElementsPluginDynamic(const std::string name, const void *data,
                           size_t length);
  ~GatherElementsPluginDynamic();
  // It doesn't make sense to make GatherElementsPluginDynamic without arguments,
  // so we delete default constructor.
  GatherElementsPluginDynamic() = delete;

  // IPluginV2DynamicExt Methods
  nvinfer1::IPluginV2DynamicExt *clone() const noexcept override;
  nvinfer1::DimsExprs getOutputDimensions(
      int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs,
      nvinfer1::IExprBuilder &exprBuilder) noexcept override;
  bool supportsFormatCombination(int pos,
                                 const nvinfer1::PluginTensorDesc *inOut,
                                 int nbInputs,
                                 int nbOutputs) noexcept override;
  void configurePlugin(const nvinfer1::DynamicPluginTensorDesc *in,
                       int nbInputs,
                       const nvinfer1::DynamicPluginTensorDesc *out,
                       int nbOutputs) noexcept override;
  size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs,
                          int nbInputs,
                          const nvinfer1::PluginTensorDesc *outputs,
                          int nbOutputs) const noexcept override;
  int enqueue(const nvinfer1::PluginTensorDesc *inputDesc,
              const nvinfer1::PluginTensorDesc *outputDesc,
              const void *const *inputs, void *const *outputs, void *workspace,
              cudaStream_t stream) noexcept override;

  // IPluginV2Ext Methods
  nvinfer1::DataType getOutputDataType(
      int index, const nvinfer1::DataType *inputTypes,
      int nbInputs) const noexcept override;

  // IPluginV2 Methods
//   const char *getPluginType() const noexcept override;
//   const char *getPluginVersion() const noexcept override;
//   int getNbOutputs() const noexcept override;
//   size_t getSerializationSize() const noexcept override;
//   void serialize(void *buffer) const noexcept override;

//  private:
//   int mDim;
  const char *getPluginType() const noexcept override;
  const char *getPluginVersion() const noexcept override;
  int getNbOutputs() const noexcept override;
  int initialize() noexcept override;
  void terminate() noexcept override;
  size_t getSerializationSize() const noexcept override;
  void serialize(void *buffer) const noexcept override;
  void destroy() noexcept override;
  void setPluginNamespace(const char *pluginNamespace) noexcept override;
  const char *getPluginNamespace() const noexcept override;

 private:
  const std::string mLayerName;
  std::string mNamespace;
  int mDim;

};

class GatherElementsPluginDynamicCreator : public nvinfer1::IPluginCreator {
 public:
  GatherElementsPluginDynamicCreator();

  const char *getPluginName() const noexcept override;

  const char *getPluginVersion() const noexcept override;

  const nvinfer1::PluginFieldCollection *getFieldNames() noexcept override;

  nvinfer1::IPluginV2 *createPlugin(const char *name,
                                    const nvinfer1::PluginFieldCollection *fc)
      noexcept override;

  nvinfer1::IPluginV2 *deserializePlugin(
      const char *name, const void *serialData,
      size_t serialLength) noexcept override;

  void setPluginNamespace(const char *pluginNamespace) noexcept override;

  const char *getPluginNamespace() const noexcept override;

 private:
  static nvinfer1::PluginFieldCollection mFC;
  static std::vector<nvinfer1::PluginField> mPluginAttributes;
  std::string mNamespace;
};

}  // namespace plugin
//}  // namespace amirstan
