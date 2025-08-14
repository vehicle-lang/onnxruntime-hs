{-# LANGUAGE CApiFFI #-}
{-# OPTIONS_GHC -Wno-unused-top-binds #-}

module Onnxruntime.CApi
  ( -- * API Base
    OrtApiVersion,
    OrtApiVersionType,
    ortApiVersion,
    ortGetApiBase,
    ortApiBaseGetVersionString,
    ortApiBaseGetApi,

    -- * API Types
    OrtApiBase,
    ExecutionMode (
      OrtSequential,
      OrtParallel
    ),
    GraphOptimizationLevel (
      OrtDisableAll,
      OrtEnableBasic,
      OrtEnableExtended,
      OrtEnableAll
    ),
    ONNXTensorElementDataType (
      ONNXTensorElementDataTypeUndefined,
      ONNXTensorElementDataTypeFloat,
      ONNXTensorElementDataTypeUint8,
      ONNXTensorElementDataTypeInt8,
      ONNXTensorElementDataTypeUint16,
      ONNXTensorElementDataTypeInt16,
      ONNXTensorElementDataTypeInt32,
      ONNXTensorElementDataTypeInt64,
      ONNXTensorElementDataTypeString,
      ONNXTensorElementDataTypeBool,
      ONNXTensorElementDataTypeFloat16,
      ONNXTensorElementDataTypeDouble,
      ONNXTensorElementDataTypeUint32,
      ONNXTensorElementDataTypeUint64,
      ONNXTensorElementDataTypeComplex64,
      ONNXTensorElementDataTypeComplex128,
      ONNXTensorElementDataTypeBfloat16,
      ONNXTensorElementDataTypeFloat8e4m3fn,
      ONNXTensorElementDataTypeFloat8e4m3fnuz,
      ONNXTensorElementDataTypeFloat8e5m2,
      ONNXTensorElementDataTypeFloat8e5m2fnuz,
      ONNXTensorElementDataTypeUint4,
      ONNXTensorElementDataTypeInt4
    ),
    ONNXType (
      ONNXTypeUnknown,
      ONNXTypeTensor,
      ONNXTypeSequence,
      ONNXTypeMap,
      ONNXTypeOpaque,
      ONNXTypeSparseTensor,
      ONNXTypeOptional
    ),
    OrtAllocatorType (
      OrtInvalidAllocator,
      OrtDeviceAllocator,
      OrtArenaAllocator
    ),
    OrtError (
      OrtError,
      ortErrorCode,
      ortErrorMessage
    ),
    OrtErrorCode (
      OrtOk,
      OrtFail,
      OrtInvalidArgument,
      OrtNoSuchfile,
      OrtNoModel,
      OrtEngineError,
      OrtRuntimeException,
      OrtInvalidProtobuf,
      OrtModelLoaded,
      OrtNotImplemented,
      OrtInvalidGraph,
      OrtEpFail
    ),
    OrtLoggingLevel (
      OrtLoggingLevelVerbose,
      OrtLoggingLevelInfo,
      OrtLoggingLevelWarning,
      OrtLoggingLevelError,
      OrtLoggingLevelFatal
    ),
    OrtMemType (
      OrtMemTypeCPUInput,
      OrtMemTypeCPUOutput,
      OrtMemTypeCPU,
      OrtMemTypeDefault
    ),
    OrtApi,
    OrtAllocator,
    OrtEnv,
    OrtMapTypeInfo,
    OrtMemoryInfo,
    OrtSession,
    OrtSessionOptions,
    OrtTensorTypeAndShapeInfo,
    OrtTypeInfo,
    OrtRunOptions,
    OrtValue,

    -- * API Functions
    ortApiGetErrorMessageAsString,
    ortApiGetErrorMessage,
    ortApiCreateEnv,
    ortApiCreateSession,
    ortApiRun,
    ortApiCreateSessionOptions,
    ortApiCloneSessionOptions,
    ortApiSetOptimizedModelFilePath,
    ortApiSetSessionExecutionMode,
    ortApiEnableProfiling,
    ortApiDisableProfiling,
    ortApiEnableMemPattern,
    ortApiDisableMemPattern,
    ortApiEnableCpuMemArena,
    ortApiDisableCpuMemArena,
    ortApiSetSessionLogId,
    ortApiSetSessionLogVerbosityLevel,
    ortApiSetSessionLogSeverityLevel,
    ortApiSetSessionGraphOptimizationLevel,
    ortApiSetIntraOpNumThreads,
    ortApiSetInterOpNumThreads,
    ortApiSessionGetInputCount,
    ortApiSessionGetOutputCount,
    ortApiSessionGetInputName,
    ortApiSessionGetOutputName,
    ortApiSessionGetInputTypeInfo,
    ortApiSessionGetOutputTypeInfo,
    ortApiCreateRunOptions,
    ortApiRunOptionsSetRunLogVerbosityLevel,
    ortApiRunOptionsSetRunLogSeverityLevel,
    ortApiRunOptionsSetRunTag,
    ortApiRunOptionsGetRunLogVerbosityLevel,
    ortApiRunOptionsGetRunLogSeverityLevel,
    ortApiRunOptionsGetRunTag,
    ortApiRunOptionsSetTerminate,
    ortApiRunOptionsUnsetTerminate,
    ortApiCreateTensorAsOrtValue,
    ortApiWithTensorWithDataAsOrtValue,
    ortApiIsTensor,
    ortApiCheckType,
    ortApiCheckTensorElementDataType,
    ortApiWithTensorData,
    ortApiCastTypeInfoToTensorInfo,
    ortApiGetOnnxTypeFromTypeInfo,
    ortApiGetTensorElementType,
    ortApiGetDimensionsCount,
    ortApiGetDimensions,
    ortApiGetTensorShapeElementCount,
    ortApiGetTensorTypeAndShape,
    ortApiGetTypeInfo,
    ortApiGetValueType,
    ortApiCreateMemoryInfo,
    ortApiCreateCpuMemoryInfo,
    ortApiGetAllocatorWithDefaultOptions,
    ortApiAddFreeDimensionOverride,
  ) where

import Control.Exception (Exception (..), assert, finally, throwIO)
import Control.Monad (unless)
import Data.ByteString (ByteString)
import Data.ByteString qualified as BS
import Data.ByteString.Char8 qualified as BSC
import Data.Coerce (coerce)
import Data.Kind (Type)
import Data.Proxy (Proxy (..))
import Data.Vector.Storable (Vector)
import Data.Vector.Storable qualified as VS
import Data.Void (Void)
import Foreign
import Foreign.C.ConstPtr.Compat (ConstPtr (..))
import Foreign.C.Types
import Foreign.C.String
import GHC.TypeLits (Natural)
import Text.Printf (printf)

#if __has_include(<onnxruntime_c_api.h>)
#include <onnxruntime_c_api.h>
#elif __has_include(<onnxruntime/onnxruntime_c_api.h>)
#include <onnxruntime/onnxruntime_c_api.h>
#elif __has_include(<onnxruntime/core/session/onnxruntime_c_api.h>)
#include <onnxruntime/core/session/onnxruntime_c_api.h>
#endif

-------------------------------------------------------------------------------
-- ONNX Runtime: API Base
-------------------------------------------------------------------------------

-------------------------------------------------------------------------------
-- OrtApiVersion

type OrtApiVersion :: Natural
type OrtApiVersion = #const ORT_API_VERSION

type OrtApiVersionType :: Type
type OrtApiVersionType = #{type uint32_t}

{- |
The API version defined in this module.

This value is used by some API functions to behave as this version of the header expects.
-}
ortApiVersion :: OrtApiVersionType
ortApiVersion = #const ORT_API_VERSION

-------------------------------------------------------------------------------
-- OrtApiBase

{- |
The helper interface to get the right version of 'OrtApi'.

Get a pointer to this structure through 'ortGetApiBase'.
-}
newtype
  {-# CTYPE "onnxruntime_c_api.h" "OrtApiBase" #-}
  OrtApiBase = OrtApiBase { ortApiBaseConstPtr :: ConstPtr OrtApiBase }

-------------------------------------------------------------------------------
-- ortApiGetBase

{- |
The Onnxruntime library's entry point to access the C API.

Call this to get the a pointer to an 'OrtApiBase'.
-}
ortGetApiBase :: IO OrtApiBase
ortGetApiBase = coerce _wrap_ortGetApiBase
{-# INLINE ortGetApiBase #-}

foreign import capi unsafe
  "onnxruntime_c_api.h OrtGetApiBase"
  _wrap_ortGetApiBase ::
    IO (ConstPtr OrtApiBase)

-------------------------------------------------------------------------------
-- OrtApiBase::GetVersionString

{- |
Returns a null terminated string of the version of the Onnxruntime library (eg: "1.8.1").
-}
ortApiBaseGetVersionString ::
  OrtApiBase ->
  IO String
ortApiBaseGetVersionString ortApiBase = do
  ConstPtr versionStringPtr <- _wrap_OrtApiBase_GetVersionString ortApiBase.ortApiBaseConstPtr
  peekCString versionStringPtr

foreign import capi unsafe
  "Onnxruntime/CApi_hsc.h _wrap_OrtApiBase_GetVersionString"
  _wrap_OrtApiBase_GetVersionString ::
    ConstPtr OrtApiBase ->
    IO (ConstPtr CChar)

#{def
  const char* _wrap_OrtApiBase_GetVersionString(const OrtApiBase* ortApiBase) {
    return ortApiBase->GetVersionString();
  }
}

-------------------------------------------------------------------------------
-- OrtApiBase::GetApi


data OrtApiUnsupportedVersionError
  = ErrOrtApiUnsupportedVersion
    -- | Requested version..
    !OrtApiVersionType
  deriving (Eq, Show)

instance Exception OrtApiUnsupportedVersionError

{- |
Get a pointer to the requested version of the 'OrtApi'
-}
ortApiBaseGetApi ::
  OrtApiBase ->
  OrtApiVersionType ->
  IO OrtApi
ortApiBaseGetApi ortApiBase version = do
  ortApi <- coerce _wrap_OrtApiBase_GetApi ortApiBase version
  if ortApi == nullPtr
    then throwIO (ErrOrtApiUnsupportedVersion version)
    else pure (coerce ortApi)
{-# INLINE ortApiBaseGetApi #-}

foreign import capi unsafe
  "Onnxruntime/CApi_hsc.h _wrap_OrtApiBase_GetApi"
  _wrap_OrtApiBase_GetApi ::
    ConstPtr OrtApiBase ->
    OrtApiVersionType ->
    IO (ConstPtr OrtApi)

#{def
  const OrtApi* _wrap_OrtApiBase_GetApi(const OrtApiBase* ortApiBase, uint32_t version) {
    return ortApiBase->GetApi(version);
  }
}


-------------------------------------------------------------------------------
-- ONNX Runtime: Primitive Types
-------------------------------------------------------------------------------

-- NOTE: This section contains those types which are passed by value.
-- NOTE: The definitions in this section are SORTED ALPHABETICALLY.

-------------------------------------------------------------------------------
-- ExecutionMode

{- |
> typedef enum ExecutionMode {
>   ORT_SEQUENTIAL = 0,
>   ORT_PARALLEL = 1,
> } ExecutionMode;
-}
newtype
  {-# CTYPE "onnxruntime_c_api.h" "ExecutionMode" #-}
  ExecutionMode = ExecutionMode
    { unExecutionMode :: #{type ExecutionMode}
    }
    deriving (Eq, Show)

pattern OrtSequential :: ExecutionMode
pattern OrtSequential = ExecutionMode ( #{const ORT_SEQUENTIAL} )

pattern OrtParallel :: ExecutionMode
pattern OrtParallel = ExecutionMode ( #{const ORT_PARALLEL} )

{-# COMPLETE
  OrtSequential,
  OrtParallel
  #-}

-------------------------------------------------------------------------------
-- GraphOptimizationLevel

{- |
> typedef enum GraphOptimizationLevel {
>   ORT_DISABLE_ALL = 0,
>   ORT_ENABLE_BASIC = 1,
>   ORT_ENABLE_EXTENDED = 2,
>   ORT_ENABLE_ALL = 99
> } GraphOptimizationLevel;
-}
newtype
  {-# CTYPE "onnxruntime_c_api.h" "GraphOptimizationLevel" #-}
  GraphOptimizationLevel = GraphOptimizationLevel
    { unGraphOptimizationLevel :: #{type GraphOptimizationLevel}
    }
    deriving (Eq, Show)


pattern OrtDisableAll :: GraphOptimizationLevel
pattern OrtDisableAll = GraphOptimizationLevel ( #{const ORT_DISABLE_ALL} )

pattern OrtEnableBasic :: GraphOptimizationLevel
pattern OrtEnableBasic = GraphOptimizationLevel ( #{const ORT_ENABLE_BASIC} )

pattern OrtEnableExtended :: GraphOptimizationLevel
pattern OrtEnableExtended = GraphOptimizationLevel ( #{const ORT_ENABLE_EXTENDED} )

pattern OrtEnableAll :: GraphOptimizationLevel
pattern OrtEnableAll = GraphOptimizationLevel ( #{const ORT_ENABLE_ALL} )

{-# COMPLETE
  OrtDisableAll,
  OrtEnableBasic,
  OrtEnableExtended,
  OrtEnableAll
  #-}

-------------------------------------------------------------------------------
-- ONNXTensorElementDataType

{- |
> typedef enum ONNXTensorElementDataType {
>   ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED,
>   ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
>   ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
>   ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,
>   ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,
>   ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,
>   ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
>   ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
>   ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING,
>   ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL,
>   ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
>   ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,
>   ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,
>   ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64,
>   ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64,
>   ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128,
>   ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16,
>   ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN,
>   ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ,
>   ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2,
>   ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ,
>   ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4,
>   ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4
> } ONNXTensorElementDataType;
-}
newtype
  {-# CTYPE "onnxruntime_c_api.h" "ONNXTensorElementDataType" #-}
  ONNXTensorElementDataType = ONNXTensorElementDataType
    { unONNXTensorElementDataType :: #{type ONNXTensorElementDataType}
    }
    deriving (Eq)

pattern ONNXTensorElementDataTypeUndefined :: ONNXTensorElementDataType
pattern ONNXTensorElementDataTypeUndefined = ONNXTensorElementDataType ( #{const ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED} )

pattern ONNXTensorElementDataTypeFloat :: ONNXTensorElementDataType
pattern ONNXTensorElementDataTypeFloat = ONNXTensorElementDataType ( #{const ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT} )

pattern ONNXTensorElementDataTypeUint8 :: ONNXTensorElementDataType
pattern ONNXTensorElementDataTypeUint8 = ONNXTensorElementDataType ( #{const ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8} )

pattern ONNXTensorElementDataTypeInt8 :: ONNXTensorElementDataType
pattern ONNXTensorElementDataTypeInt8 = ONNXTensorElementDataType ( #{const ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8} )

pattern ONNXTensorElementDataTypeUint16 :: ONNXTensorElementDataType
pattern ONNXTensorElementDataTypeUint16 = ONNXTensorElementDataType ( #{const ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16} )

pattern ONNXTensorElementDataTypeInt16 :: ONNXTensorElementDataType
pattern ONNXTensorElementDataTypeInt16 = ONNXTensorElementDataType ( #{const ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16} )

pattern ONNXTensorElementDataTypeInt32 :: ONNXTensorElementDataType
pattern ONNXTensorElementDataTypeInt32 = ONNXTensorElementDataType ( #{const ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32} )

pattern ONNXTensorElementDataTypeInt64 :: ONNXTensorElementDataType
pattern ONNXTensorElementDataTypeInt64 = ONNXTensorElementDataType ( #{const ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64} )

pattern ONNXTensorElementDataTypeString :: ONNXTensorElementDataType
pattern ONNXTensorElementDataTypeString = ONNXTensorElementDataType ( #{const ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING} )

pattern ONNXTensorElementDataTypeBool :: ONNXTensorElementDataType
pattern ONNXTensorElementDataTypeBool = ONNXTensorElementDataType ( #{const ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL} )

pattern ONNXTensorElementDataTypeFloat16 :: ONNXTensorElementDataType
pattern ONNXTensorElementDataTypeFloat16 = ONNXTensorElementDataType ( #{const ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16} )

pattern ONNXTensorElementDataTypeDouble :: ONNXTensorElementDataType
pattern ONNXTensorElementDataTypeDouble = ONNXTensorElementDataType ( #{const ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE} )

pattern ONNXTensorElementDataTypeUint32 :: ONNXTensorElementDataType
pattern ONNXTensorElementDataTypeUint32 = ONNXTensorElementDataType ( #{const ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32} )

pattern ONNXTensorElementDataTypeUint64 :: ONNXTensorElementDataType
pattern ONNXTensorElementDataTypeUint64 = ONNXTensorElementDataType ( #{const ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64} )

pattern ONNXTensorElementDataTypeComplex64 :: ONNXTensorElementDataType
pattern ONNXTensorElementDataTypeComplex64 = ONNXTensorElementDataType ( #{const ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64} )

pattern ONNXTensorElementDataTypeComplex128 :: ONNXTensorElementDataType
pattern ONNXTensorElementDataTypeComplex128 = ONNXTensorElementDataType ( #{const ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128} )

pattern ONNXTensorElementDataTypeBfloat16 :: ONNXTensorElementDataType
pattern ONNXTensorElementDataTypeBfloat16 = ONNXTensorElementDataType ( #{const ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16} )

pattern ONNXTensorElementDataTypeFloat8e4m3fn :: ONNXTensorElementDataType
pattern ONNXTensorElementDataTypeFloat8e4m3fn = ONNXTensorElementDataType ( #{const ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN} )

pattern ONNXTensorElementDataTypeFloat8e4m3fnuz :: ONNXTensorElementDataType
pattern ONNXTensorElementDataTypeFloat8e4m3fnuz = ONNXTensorElementDataType ( #{const ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ} )

pattern ONNXTensorElementDataTypeFloat8e5m2 :: ONNXTensorElementDataType
pattern ONNXTensorElementDataTypeFloat8e5m2 = ONNXTensorElementDataType ( #{const ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2} )

pattern ONNXTensorElementDataTypeFloat8e5m2fnuz :: ONNXTensorElementDataType
pattern ONNXTensorElementDataTypeFloat8e5m2fnuz = ONNXTensorElementDataType ( #{const ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ} )

pattern ONNXTensorElementDataTypeUint4 :: ONNXTensorElementDataType
pattern ONNXTensorElementDataTypeUint4 = ONNXTensorElementDataType ( #{const ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT4} )

pattern ONNXTensorElementDataTypeInt4 :: ONNXTensorElementDataType
pattern ONNXTensorElementDataTypeInt4 = ONNXTensorElementDataType ( #{const ONNX_TENSOR_ELEMENT_DATA_TYPE_INT4} )

{-# COMPLETE
  ONNXTensorElementDataTypeUndefined,
  ONNXTensorElementDataTypeFloat,
  ONNXTensorElementDataTypeUint8,
  ONNXTensorElementDataTypeInt8,
  ONNXTensorElementDataTypeUint16,
  ONNXTensorElementDataTypeInt16,
  ONNXTensorElementDataTypeInt32,
  ONNXTensorElementDataTypeInt64,
  ONNXTensorElementDataTypeString,
  ONNXTensorElementDataTypeBool,
  ONNXTensorElementDataTypeFloat16,
  ONNXTensorElementDataTypeDouble,
  ONNXTensorElementDataTypeUint32,
  ONNXTensorElementDataTypeUint64,
  ONNXTensorElementDataTypeComplex64,
  ONNXTensorElementDataTypeComplex128,
  ONNXTensorElementDataTypeBfloat16,
  ONNXTensorElementDataTypeFloat8e4m3fn,
  ONNXTensorElementDataTypeFloat8e4m3fnuz,
  ONNXTensorElementDataTypeFloat8e5m2,
  ONNXTensorElementDataTypeFloat8e5m2fnuz,
  ONNXTensorElementDataTypeUint4,
  ONNXTensorElementDataTypeInt4
  #-}

instance Show ONNXTensorElementDataType where
  show = \case
    ONNXTensorElementDataTypeUndefined -> "ONNXTensorElementDataTypeUndefined"
    ONNXTensorElementDataTypeFloat -> "ONNXTensorElementDataTypeFloat"
    ONNXTensorElementDataTypeUint8 -> "ONNXTensorElementDataTypeUint8"
    ONNXTensorElementDataTypeInt8 -> "ONNXTensorElementDataTypeInt8"
    ONNXTensorElementDataTypeUint16 -> "ONNXTensorElementDataTypeUint16"
    ONNXTensorElementDataTypeInt16 -> "ONNXTensorElementDataTypeInt16"
    ONNXTensorElementDataTypeInt32 -> "ONNXTensorElementDataTypeInt32"
    ONNXTensorElementDataTypeInt64 -> "ONNXTensorElementDataTypeInt64"
    ONNXTensorElementDataTypeString -> "ONNXTensorElementDataTypeString"
    ONNXTensorElementDataTypeBool -> "ONNXTensorElementDataTypeBool"
    ONNXTensorElementDataTypeFloat16 -> "ONNXTensorElementDataTypeFloat16"
    ONNXTensorElementDataTypeDouble -> "ONNXTensorElementDataTypeDouble"
    ONNXTensorElementDataTypeUint32 -> "ONNXTensorElementDataTypeUint32"
    ONNXTensorElementDataTypeUint64 -> "ONNXTensorElementDataTypeUint64"
    ONNXTensorElementDataTypeComplex64 -> "ONNXTensorElementDataTypeComplex64"
    ONNXTensorElementDataTypeComplex128 -> "ONNXTensorElementDataTypeComplex128"
    ONNXTensorElementDataTypeBfloat16 -> "ONNXTensorElementDataTypeBfloat16"
    ONNXTensorElementDataTypeFloat8e4m3fn -> "ONNXTensorElementDataTypeFloat8e4m3fn"
    ONNXTensorElementDataTypeFloat8e4m3fnuz -> "ONNXTensorElementDataTypeFloat8e4m3fnuz"
    ONNXTensorElementDataTypeFloat8e5m2 -> "ONNXTensorElementDataTypeFloat8e5m2"
    ONNXTensorElementDataTypeFloat8e5m2fnuz -> "ONNXTensorElementDataTypeFloat8e5m2fnuz"
    ONNXTensorElementDataTypeUint4 -> "ONNXTensorElementDataTypeUint4"
    ONNXTensorElementDataTypeInt4 -> "ONNXTensorElementDataTypeInt4"

class Storable a => IsONNXTensorElementDataType a where
  getONNXTensorElementDataType :: Proxy a -> ONNXTensorElementDataType

instance IsONNXTensorElementDataType Float where
  getONNXTensorElementDataType _ = ONNXTensorElementDataTypeFloat

instance IsONNXTensorElementDataType Double where
  getONNXTensorElementDataType _ = ONNXTensorElementDataTypeDouble

instance IsONNXTensorElementDataType Int8 where
  getONNXTensorElementDataType _ = ONNXTensorElementDataTypeInt8

instance IsONNXTensorElementDataType Int16 where
  getONNXTensorElementDataType _ = ONNXTensorElementDataTypeInt16

instance IsONNXTensorElementDataType Int32 where
  getONNXTensorElementDataType _ = ONNXTensorElementDataTypeInt32

instance IsONNXTensorElementDataType Int64 where
  getONNXTensorElementDataType _ = ONNXTensorElementDataTypeInt64

instance IsONNXTensorElementDataType Word8 where
  getONNXTensorElementDataType _ = ONNXTensorElementDataTypeUint8

instance IsONNXTensorElementDataType Word16 where
  getONNXTensorElementDataType _ = ONNXTensorElementDataTypeUint16

instance IsONNXTensorElementDataType Word32 where
  getONNXTensorElementDataType _ = ONNXTensorElementDataTypeUint32

instance IsONNXTensorElementDataType Word64 where
  getONNXTensorElementDataType _ = ONNXTensorElementDataTypeUint64

-- NOTE: The following 'ONNXTensorElementDataType' types are unsupported:
--
-- [@ONNXTensorElementDataTypeUndefined@]:
--   Unsupported as input format for obvious reasons.
-- [@ONNXTensorElementDataTypeString@]:
--   Maps to C++ type std::string
-- [@ONNXTensorElementDataTypeFloat16@]:
--   Maps to float16_t
-- [@ONNXTensorElementDataTypeComplex64@]:
--   Maps to C++ type std::complex<float32>
-- [@ONNXTensorElementDataTypeComplex128@]:
--   Maps to C++ type std::complex<float64>
-- [@ONNXTensorElementDataTypeBfloat16@]:
--   Maps to non-IEEE floating-point format based on IEEE754 single-precision.
-- [@ONNXTensorElementDataTypeFloat8e4m3fn@]:
--   Maps to non-IEEE floating-point format based on IEEE754 single-precision.
-- [@ONNXTensorElementDataTypeFloat8e4m3fnuz@]:
--   Maps to non-IEEE floating-point format based on IEEE754 single-precision.
-- [@ONNXTensorElementDataTypeFloat8e5m2@]:
--   Maps to non-IEEE floating-point format based on IEEE754 single-precision.
-- [@ONNXTensorElementDataTypeFloat8e5m2fnuz@]:
--   Maps to non-IEEE floating-point format based on IEEE754 single-precision.
-- [@ONNXTensorElementDataTypeUint4@]:
--   Maps to a pair of packed uint4 values.
-- [@ONNXTensorElementDataTypeInt4@]:
--   Maps to a pair of packed int4 values.

-- | Type-level tag for supported 'ONNXTensorElementDataType' types.
type data ONNXTensorElementDataTypeTag
  = ONNXTensorElementDataTypeTagFloat
  | ONNXTensorElementDataTypeTagDouble
  | ONNXTensorElementDataTypeTagInt8
  | ONNXTensorElementDataTypeTagInt16
  | ONNXTensorElementDataTypeTagInt32
  | ONNXTensorElementDataTypeTagInt64
  | ONNXTensorElementDataTypeTagUint8
  | ONNXTensorElementDataTypeTagUint16
  | ONNXTensorElementDataTypeTagUint32
  | ONNXTensorElementDataTypeTagUint64

-------------------------------------------------------------------------------
-- ONNXType

{-
> typedef enum ONNXType {
>   ONNX_TYPE_UNKNOWN,
>   ONNX_TYPE_TENSOR,
>   ONNX_TYPE_SEQUENCE,
>   ONNX_TYPE_MAP,
>   ONNX_TYPE_OPAQUE,
>   ONNX_TYPE_SPARSETENSOR,
>   ONNX_TYPE_OPTIONAL
> } ONNXType;
-}
newtype
  {-# CTYPE "onnxruntime_c_api.h" "ONNXType" #-}
  ONNXType = ONNXType
    { unONNXType :: #{type ONNXType}
    }
    deriving (Eq)

pattern ONNXTypeUnknown :: ONNXType
pattern ONNXTypeUnknown = ONNXType ( #{const ONNX_TYPE_UNKNOWN} )

pattern ONNXTypeTensor :: ONNXType
pattern ONNXTypeTensor = ONNXType ( #{const ONNX_TYPE_TENSOR} )

pattern ONNXTypeSequence :: ONNXType
pattern ONNXTypeSequence = ONNXType ( #{const ONNX_TYPE_SEQUENCE} )

pattern ONNXTypeMap :: ONNXType
pattern ONNXTypeMap = ONNXType ( #{const ONNX_TYPE_MAP} )

pattern ONNXTypeOpaque :: ONNXType
pattern ONNXTypeOpaque = ONNXType ( #{const ONNX_TYPE_OPAQUE} )

pattern ONNXTypeSparseTensor :: ONNXType
pattern ONNXTypeSparseTensor = ONNXType ( #{const ONNX_TYPE_SPARSETENSOR} )

pattern ONNXTypeOptional :: ONNXType
pattern ONNXTypeOptional = ONNXType ( #{const ONNX_TYPE_OPTIONAL} )

{-# COMPLETE
  ONNXTypeUnknown,
  ONNXTypeTensor,
  ONNXTypeSequence,
  ONNXTypeMap,
  ONNXTypeOpaque,
  ONNXTypeSparseTensor,
  ONNXTypeOptional
  #-}

instance Show ONNXType where
  show = \case
    ONNXTypeUnknown -> "ONNXTypeUnknown"
    ONNXTypeTensor -> "ONNXTypeTensor"
    ONNXTypeSequence -> "ONNXTypeSequence"
    ONNXTypeMap -> "ONNXTypeMap"
    ONNXTypeOpaque -> "ONNXTypeOpaque"
    ONNXTypeSparseTensor -> "ONNXTypeSparseTensor"
    ONNXTypeOptional -> "ONNXTypeOptional"

-- | Type-level tag for supported 'ONNXType' types.
type data ONNXTypeTag
  = ONNXTypeTagTensor ONNXTensorElementDataTypeTag

-------------------------------------------------------------------------------
-- OrtAllocatorType

{- |
> typedef enum OrtAllocatorType {
>   OrtInvalidAllocator = -1,
>   OrtDeviceAllocator = 0,
>   OrtArenaAllocator = 1
> } OrtAllocatorType;
-}
newtype
  {-# CTYPE "onnxruntime_c_api.h" "OrtAllocatorType" #-}
  OrtAllocatorType = OrtAllocatorType
    { unOrtAllocatorType :: #{type OrtAllocatorType}
    }
    deriving (Eq, Show)

pattern OrtInvalidAllocator :: OrtAllocatorType
pattern OrtInvalidAllocator = OrtAllocatorType ( #{const OrtInvalidAllocator} )

pattern OrtDeviceAllocator :: OrtAllocatorType
pattern OrtDeviceAllocator = OrtAllocatorType ( #{const OrtDeviceAllocator} )

pattern OrtArenaAllocator :: OrtAllocatorType
pattern OrtArenaAllocator = OrtAllocatorType ( #{const OrtArenaAllocator} )

{-# COMPLETE
  OrtInvalidAllocator,
  OrtDeviceAllocator,
  OrtArenaAllocator
  #-}

-------------------------------------------------------------------------------
-- OrtErrorCode

{- |
> typedef enum OrtErrorCode {
>   ORT_OK,
>   ORT_FAIL,
>   ORT_INVALID_ARGUMENT,
>   ORT_NO_SUCHFILE,
>   ORT_NO_MODEL,
>   ORT_ENGINE_ERROR,
>   ORT_RUNTIME_EXCEPTION,
>   ORT_INVALID_PROTOBUF,
>   ORT_MODEL_LOADED,
>   ORT_NOT_IMPLEMENTED,
>   ORT_INVALID_GRAPH,
>   ORT_EP_FAIL,
> } OrtErrorCode;
-}
newtype
  {-# CTYPE "onnxruntime_c_api.h" "OrtErrorCode" #-}
  OrtErrorCode = OrtErrorCode
    { unOrtErrorCode :: #{type OrtErrorCode}
    }
    deriving (Eq, Show)

pattern OrtOk :: OrtErrorCode
pattern OrtOk = OrtErrorCode ( #{const ORT_OK} )

pattern OrtFail :: OrtErrorCode
pattern OrtFail = OrtErrorCode ( #{const ORT_FAIL} )

pattern OrtInvalidArgument :: OrtErrorCode
pattern OrtInvalidArgument = OrtErrorCode ( #{const ORT_INVALID_ARGUMENT} )

pattern OrtNoSuchfile :: OrtErrorCode
pattern OrtNoSuchfile = OrtErrorCode ( #{const ORT_NO_SUCHFILE} )

pattern OrtNoModel :: OrtErrorCode
pattern OrtNoModel = OrtErrorCode ( #{const ORT_NO_MODEL} )

pattern OrtEngineError :: OrtErrorCode
pattern OrtEngineError = OrtErrorCode ( #{const ORT_ENGINE_ERROR} )

pattern OrtRuntimeException :: OrtErrorCode
pattern OrtRuntimeException = OrtErrorCode ( #{const ORT_RUNTIME_EXCEPTION} )

pattern OrtInvalidProtobuf :: OrtErrorCode
pattern OrtInvalidProtobuf = OrtErrorCode ( #{const ORT_INVALID_PROTOBUF} )

pattern OrtModelLoaded :: OrtErrorCode
pattern OrtModelLoaded = OrtErrorCode ( #{const ORT_MODEL_LOADED} )

pattern OrtNotImplemented :: OrtErrorCode
pattern OrtNotImplemented = OrtErrorCode ( #{const ORT_NOT_IMPLEMENTED} )

pattern OrtInvalidGraph :: OrtErrorCode
pattern OrtInvalidGraph = OrtErrorCode ( #{const ORT_INVALID_GRAPH} )

pattern OrtEpFail :: OrtErrorCode
pattern OrtEpFail = OrtErrorCode ( #{const ORT_EP_FAIL} )

{-# COMPLETE
  OrtOk,
  OrtFail,
  OrtInvalidArgument,
  OrtNoSuchfile,
  OrtNoModel,
  OrtEngineError,
  OrtRuntimeException,
  OrtInvalidProtobuf,
  OrtModelLoaded,
  OrtNotImplemented,
  OrtInvalidGraph,
  OrtEpFail
  #-}

instance Exception OrtErrorCode where
  displayException = \case
    OrtOk -> "ORT_OK"
    OrtFail -> "ORT_FAIL"
    OrtInvalidArgument -> "ORT_INVALID_ARGUMENT"
    OrtNoSuchfile -> "ORT_NO_SUCHFILE"
    OrtNoModel -> "ORT_NO_MODEL"
    OrtEngineError -> "ORT_ENGINE_ERROR"
    OrtRuntimeException -> "ORT_RUNTIME_EXCEPTION"
    OrtInvalidProtobuf -> "ORT_INVALID_PROTOBUF"
    OrtModelLoaded -> "ORT_MODEL_LOADED"
    OrtNotImplemented -> "ORT_NOT_IMPLEMENTED"
    OrtInvalidGraph -> "ORT_INVALID_GRAPH"
    OrtEpFail -> "ORT_EP_FAIL"

-------------------------------------------------------------------------------
-- OrtLoggingLevel

{-|
> typedef enum OrtLoggingLevel {
>   ORT_LOGGING_LEVEL_VERBOSE,  ///< Verbose informational messages (least severe).
>   ORT_LOGGING_LEVEL_INFO,     ///< Informational messages.
>   ORT_LOGGING_LEVEL_WARNING,  ///< Warning messages.
>   ORT_LOGGING_LEVEL_ERROR,    ///< Error messages.
>   ORT_LOGGING_LEVEL_FATAL,    ///< Fatal error messages (most severe).
> } OrtLoggingLevel;
-}
newtype
  {-# CTYPE "onnxruntime_c_api.h" "OrtLoggingLevel" #-}
  OrtLoggingLevel = OrtLoggingLevel
    { unOrtLoggingLevel :: #{type OrtLoggingLevel}
    }
    deriving (Eq, Show)

pattern OrtLoggingLevelVerbose :: OrtLoggingLevel
pattern OrtLoggingLevelVerbose = OrtLoggingLevel ( #{const ORT_LOGGING_LEVEL_VERBOSE} )

pattern OrtLoggingLevelInfo :: OrtLoggingLevel
pattern OrtLoggingLevelInfo = OrtLoggingLevel ( #{const ORT_LOGGING_LEVEL_INFO} )

pattern OrtLoggingLevelWarning :: OrtLoggingLevel
pattern OrtLoggingLevelWarning = OrtLoggingLevel ( #{const ORT_LOGGING_LEVEL_WARNING} )

pattern OrtLoggingLevelError :: OrtLoggingLevel
pattern OrtLoggingLevelError = OrtLoggingLevel ( #{const ORT_LOGGING_LEVEL_ERROR} )

pattern OrtLoggingLevelFatal :: OrtLoggingLevel
pattern OrtLoggingLevelFatal = OrtLoggingLevel ( #{const ORT_LOGGING_LEVEL_FATAL} )

{-# COMPLETE
  OrtLoggingLevelVerbose,
  OrtLoggingLevelInfo,
  OrtLoggingLevelWarning,
  OrtLoggingLevelError,
  OrtLoggingLevelFatal
  #-}

-------------------------------------------------------------------------------
-- OrtMemType

{- |
> typedef enum OrtMemType {
>   OrtMemTypeCPUInput = -2,
>   OrtMemTypeCPUOutput = -1,
>   OrtMemTypeCPU = OrtMemTypeCPUOutput,
>   OrtMemTypeDefault = 0,
> } OrtMemType;
-}
newtype
  {-# CTYPE "onnxruntime_c_api.h" "OrtMemType" #-}
  OrtMemType = OrtMemType
    { unOrtMemType :: #{type OrtMemType}
    }
    deriving (Eq, Show)

pattern OrtMemTypeCPUInput :: OrtMemType
pattern OrtMemTypeCPUInput = OrtMemType ( #{const OrtMemTypeCPUInput} )

pattern OrtMemTypeCPUOutput :: OrtMemType
pattern OrtMemTypeCPUOutput = OrtMemType ( #{const OrtMemTypeCPUOutput} )

pattern OrtMemTypeCPU :: OrtMemType
pattern OrtMemTypeCPU = OrtMemType ( #{const OrtMemTypeCPU} )

pattern OrtMemTypeDefault :: OrtMemType
pattern OrtMemTypeDefault = OrtMemType ( #{const OrtMemTypeDefault} )

{-# COMPLETE
  OrtMemTypeCPUInput,
  OrtMemTypeCPUOutput,
  OrtMemTypeCPU,
  OrtMemTypeDefault
  #-}

-------------------------------------------------------------------------------
-- ONNX Runtime: Types
-------------------------------------------------------------------------------

-- NOTE: This section contains those types which are passed by reference.
-- NOTE: The definitions in this section are SORTED ALPHABETICALLY.

-------------------------------------------------------------------------------
-- OrtApi

newtype
  {-# CTYPE "onnxruntime_c_api.h" "OrtApi" #-}
  OrtApi = OrtApi { ortApiConstPtr :: ConstPtr OrtApi }

class HasOrtApi a where
  type CType a
  getOrtApi :: a -> IO OrtApi
  withCTypePtr :: a -> (Ptr (CType a) -> IO b) -> IO b

-- | Marshall a list of ONNX Runtime types as an array of pointers.
withCTypeArrayLen ::
  (HasOrtApi a) =>
  [a] ->
  (Int -> Ptr (Ptr (CType a)) -> IO b) ->
  IO b
withCTypeArrayLen = withArrayLenWith withCTypePtr

-- | Internal helper.
withCStringArrayLen ::
  [String] ->
  (Int -> Ptr CString -> IO a) ->
  IO a
withCStringArrayLen = withArrayLenWith withCString

-- | Internal helper.
withArrayLenWith ::
  (forall c. a -> (Ptr b -> IO c) -> IO c) ->
  [a] ->
  (Int -> Ptr (Ptr b) -> IO r) ->
  IO r
withArrayLenWith withPtr xs action = go xs []
  where
  go [] acc = withArrayLen (reverse acc) action
  go (y : ys) acc = withPtr y (\yPtr -> go ys (yPtr : acc))

-------------------------------------------------------------------------------
-- OrtAllocator

data
  {-# CTYPE "onnxruntime_c_api.h" "COrtAllocator" #-}
  COrtAllocator

#{def
  typedef OrtAllocator COrtAllocator;
}

newtype
  {-# CTYPE "Onnxruntime/CApi_hsc.h" "HsOrtAllocator" #-}
  OrtAllocator = OrtAllocator { ortAllocatorForeignPtr :: ForeignPtr OrtAllocator }

#{def
  typedef struct HsOrtAllocator {
    const OrtApi* ortApi;
    COrtAllocator* ortAllocator;
  } HsOrtAllocator;
}

instance HasOrtApi OrtAllocator where
  type CType OrtAllocator = COrtAllocator
  getOrtApi ortAllocator =
    withOrtAllocatorPtr ortAllocator $ \ortAllocatorPtr ->
      OrtApi <$> #{peek HsOrtAllocator, ortApi} ortAllocatorPtr
  withCTypePtr = withCOrtAllocatorPtr

-- | Internal helper.
withOrtAllocatorPtr ::
  OrtAllocator ->
  (Ptr OrtAllocator -> IO a) ->
  IO a
withOrtAllocatorPtr ortAllocator =
  withForeignPtr ortAllocator.ortAllocatorForeignPtr

-- | Internal helper.
withCOrtAllocatorPtr ::
  OrtAllocator ->
  (Ptr COrtAllocator -> IO a) ->
  IO a
withCOrtAllocatorPtr ortAllocator action =
  withOrtAllocatorPtr ortAllocator $ \ortAllocatorPtr -> do
    cOrtAllocatorPtr <- #{peek HsOrtAllocator, ortAllocator} ortAllocatorPtr
    action cOrtAllocatorPtr

-- | Internal helper.
wrapCOrtAllocator ::
  OrtApi ->
  Ptr COrtAllocator ->
  IO OrtAllocator
wrapCOrtAllocator ortApi rawOrtAllocatorPtr = do
  ortAllocatorPtr <- _wrap_COrtAllocator ortApi.ortApiConstPtr rawOrtAllocatorPtr
  ortAllocatorForeignPtr <- newForeignPtr _wrap_OrtApi_ReleaseAllocator ortAllocatorPtr
  pure $ OrtAllocator ortAllocatorForeignPtr

foreign import capi unsafe
  "Onnxruntime/CApi_hsc.h _wrap_COrtAllocator"
  _wrap_COrtAllocator ::
    ConstPtr OrtApi ->
    Ptr COrtAllocator ->
    IO (Ptr OrtAllocator)

#{def
  HsOrtAllocator* _wrap_COrtAllocator(
    const OrtApi* ortApi,
    COrtAllocator* ortAllocator
  ) {
    HsOrtAllocator *out = malloc(sizeof *out);
    out->ortApi = ortApi;
    out->ortAllocator = ortAllocator;
    return out;
  }
}

foreign import capi unsafe
  "Onnxruntime/CApi_hsc.h &_wrap_OrtApi_ReleaseAllocator"
  _wrap_OrtApi_ReleaseAllocator ::
    FunPtr (
      Ptr OrtAllocator ->
      IO ()
    )

#{def
  void _wrap_OrtApi_ReleaseAllocator(HsOrtAllocator* ortAllocator) {
    ortAllocator->ortApi->ReleaseAllocator(ortAllocator->ortAllocator);
    free(ortAllocator);
  }
}

-------------------------------------------------------------------------------
-- OrtEnv

data
  {-# CTYPE "onnxruntime_c_api.h" "COrtEnv" #-}
  COrtEnv

#{def
  typedef OrtEnv COrtEnv;
}

newtype
  {-# CTYPE "Onnxruntime/CApi_hsc.h" "HsOrtEnv" #-}
  OrtEnv = OrtEnv { ortEnvForeignPtr :: ForeignPtr OrtEnv }

#{def
  typedef struct HsOrtEnv {
    const OrtApi* ortApi;
    COrtEnv* ortEnv;
  } HsOrtEnv;
}

instance HasOrtApi OrtEnv where
  type CType OrtEnv = COrtEnv
  getOrtApi ortEnv =
    withOrtEnvPtr ortEnv $ \ortEnvPtr ->
      OrtApi <$> #{peek HsOrtEnv, ortApi} ortEnvPtr
  withCTypePtr = withCOrtEnvPtr

-- | Internal helper.
withOrtEnvPtr ::
  OrtEnv ->
  (Ptr OrtEnv -> IO a) ->
  IO a
withOrtEnvPtr ortEnv =
  withForeignPtr ortEnv.ortEnvForeignPtr

-- | Internal helper.
withCOrtEnvPtr ::
  OrtEnv ->
  (Ptr COrtEnv -> IO a) ->
  IO a
withCOrtEnvPtr ortEnv action =
  withOrtEnvPtr ortEnv $ \ortEnvPtr -> do
    cOrtEnvPtr <- #{peek HsOrtEnv, ortEnv} ortEnvPtr
    action cOrtEnvPtr

-- | Internal helper.
wrapCOrtEnv ::
  OrtApi ->
  Ptr COrtEnv ->
  IO OrtEnv
wrapCOrtEnv ortApi rawOrtEnvPtr = do
  ortEnvPtr <- _wrap_COrtEnv ortApi.ortApiConstPtr rawOrtEnvPtr
  ortEnvForeignPtr <- newForeignPtr _wrap_OrtApi_ReleaseEnv ortEnvPtr
  pure $ OrtEnv ortEnvForeignPtr

foreign import capi unsafe
  "Onnxruntime/CApi_hsc.h _wrap_COrtEnv"
  _wrap_COrtEnv ::
    ConstPtr OrtApi ->
    Ptr COrtEnv ->
    IO (Ptr OrtEnv)

#{def
  HsOrtEnv* _wrap_COrtEnv(
    const OrtApi* ortApi,
    COrtEnv* ortEnv
  ) {
    HsOrtEnv *out = malloc(sizeof *out);
    out->ortApi = ortApi;
    out->ortEnv = ortEnv;
    return out;
  }
}

foreign import capi unsafe
  "Onnxruntime/CApi_hsc.h &_wrap_OrtApi_ReleaseEnv"
  _wrap_OrtApi_ReleaseEnv ::
    FunPtr (
      Ptr OrtEnv ->
      IO ()
    )

#{def
  void _wrap_OrtApi_ReleaseEnv(HsOrtEnv* ortEnv) {
    ortEnv->ortApi->ReleaseEnv(ortEnv->ortEnv);
    free(ortEnv);
  }
}

-------------------------------------------------------------------------------
-- OrtMapTypeInfo

data
  {-# CTYPE "onnxruntime_c_api.h" "COrtMapTypeInfo" #-}
  COrtMapTypeInfo

#{def
  typedef OrtMapTypeInfo COrtMapTypeInfo;
}

newtype
  {-# CTYPE "Onnxruntime/CApi_hsc.h" "HsOrtMapTypeInfo" #-}
  OrtMapTypeInfo = OrtMapTypeInfo { ortMapTypeInfoForeignPtr :: ForeignPtr OrtMapTypeInfo }

#{def
  typedef struct HsOrtMapTypeInfo {
    const OrtApi* ortApi;
    COrtMapTypeInfo* ortMapTypeInfo;
  } HsOrtMapTypeInfo;
}

instance HasOrtApi OrtMapTypeInfo where
  type CType OrtMapTypeInfo = COrtMapTypeInfo
  getOrtApi ortMapTypeInfo =
    withOrtMapTypeInfoPtr ortMapTypeInfo $ \ortMapTypeInfoPtr ->
      OrtApi <$> #{peek HsOrtMapTypeInfo, ortApi} ortMapTypeInfoPtr
  withCTypePtr = withCOrtMapTypeInfoPtr

-- | Internal helper.
withOrtMapTypeInfoPtr ::
  OrtMapTypeInfo ->
  (Ptr OrtMapTypeInfo -> IO a) ->
  IO a
withOrtMapTypeInfoPtr ortMapTypeInfo =
  withForeignPtr ortMapTypeInfo.ortMapTypeInfoForeignPtr

-- | Internal helper.
withCOrtMapTypeInfoPtr ::
  OrtMapTypeInfo ->
  (Ptr COrtMapTypeInfo -> IO a) ->
  IO a
withCOrtMapTypeInfoPtr ortMapTypeInfo action =
  withOrtMapTypeInfoPtr ortMapTypeInfo $ \ortMapTypeInfoPtr -> do
    cOrtMapTypeInfoPtr <- #{peek HsOrtMapTypeInfo, ortMapTypeInfo} ortMapTypeInfoPtr
    action cOrtMapTypeInfoPtr

-- | Internal helper.
wrapCOrtMapTypeInfo ::
  OrtApi ->
  Ptr COrtMapTypeInfo ->
  IO OrtMapTypeInfo
wrapCOrtMapTypeInfo ortApi rawOrtMapTypeInfoPtr = do
  ortMapTypeInfoPtr <- _wrap_COrtMapTypeInfo ortApi.ortApiConstPtr rawOrtMapTypeInfoPtr
  ortMapTypeInfoForeignPtr <- newForeignPtr _wrap_OrtApi_ReleaseMapTypeInfo ortMapTypeInfoPtr
  pure $ OrtMapTypeInfo ortMapTypeInfoForeignPtr

foreign import capi unsafe
  "Onnxruntime/CApi_hsc.h _wrap_COrtMapTypeInfo"
  _wrap_COrtMapTypeInfo ::
    ConstPtr OrtApi ->
    Ptr COrtMapTypeInfo ->
    IO (Ptr OrtMapTypeInfo)

#{def
  HsOrtMapTypeInfo* _wrap_COrtMapTypeInfo(
    const OrtApi* ortApi,
    COrtMapTypeInfo* ortMapTypeInfo
  ) {
    HsOrtMapTypeInfo *out = malloc(sizeof *out);
    out->ortApi = ortApi;
    out->ortMapTypeInfo = ortMapTypeInfo;
    return out;
  }
}

foreign import capi unsafe
  "Onnxruntime/CApi_hsc.h &_wrap_OrtApi_ReleaseMapTypeInfo"
  _wrap_OrtApi_ReleaseMapTypeInfo ::
    FunPtr (
      Ptr OrtMapTypeInfo ->
      IO ()
    )

#{def
  void _wrap_OrtApi_ReleaseMapTypeInfo(HsOrtMapTypeInfo* ortMapTypeInfo) {
    ortMapTypeInfo->ortApi->ReleaseMapTypeInfo(ortMapTypeInfo->ortMapTypeInfo);
    free(ortMapTypeInfo);
  }
}

-------------------------------------------------------------------------------
-- OrtMemoryInfo

data
  {-# CTYPE "onnxruntime_c_api.h" "COrtMemoryInfo" #-}
  COrtMemoryInfo

#{def
  typedef OrtMemoryInfo COrtMemoryInfo;
}

newtype
  {-# CTYPE "Onnxruntime/CApi_hsc.h" "HsOrtMemoryInfo" #-}
  OrtMemoryInfo = OrtMemoryInfo { ortMemoryInfoForeignPtr :: ForeignPtr OrtMemoryInfo }

#{def
  typedef struct HsOrtMemoryInfo {
    const OrtApi* ortApi;
    COrtMemoryInfo* ortMemoryInfo;
  } HsOrtMemoryInfo;
}

instance HasOrtApi OrtMemoryInfo where
  type CType OrtMemoryInfo = COrtMemoryInfo
  getOrtApi ortMemoryInfo =
    withOrtMemoryInfoPtr ortMemoryInfo $ \ortMemoryInfoPtr ->
      OrtApi <$> #{peek HsOrtMemoryInfo, ortApi} ortMemoryInfoPtr
  withCTypePtr = withCOrtMemoryInfoPtr

-- | Internal helper.
withOrtMemoryInfoPtr ::
  OrtMemoryInfo ->
  (Ptr OrtMemoryInfo -> IO a) ->
  IO a
withOrtMemoryInfoPtr ortMemoryInfo =
  withForeignPtr ortMemoryInfo.ortMemoryInfoForeignPtr

-- | Internal helper.
withCOrtMemoryInfoPtr ::
  OrtMemoryInfo ->
  (Ptr COrtMemoryInfo -> IO a) ->
  IO a
withCOrtMemoryInfoPtr ortMemoryInfo action =
  withOrtMemoryInfoPtr ortMemoryInfo $ \ortMemoryInfoPtr -> do
    cOrtMemoryInfoPtr <- #{peek HsOrtMemoryInfo, ortMemoryInfo} ortMemoryInfoPtr
    action cOrtMemoryInfoPtr

-- | Internal helper.
wrapCOrtMemoryInfo ::
  OrtApi ->
  Ptr COrtMemoryInfo ->
  IO OrtMemoryInfo
wrapCOrtMemoryInfo ortApi rawOrtMemoryInfoPtr = do
  ortMemoryInfoPtr <- _wrap_COrtMemoryInfo ortApi.ortApiConstPtr rawOrtMemoryInfoPtr
  ortMemoryInfoForeignPtr <- newForeignPtr _wrap_OrtApi_ReleaseMemoryInfo ortMemoryInfoPtr
  pure $ OrtMemoryInfo ortMemoryInfoForeignPtr

foreign import capi unsafe
  "Onnxruntime/CApi_hsc.h _wrap_COrtMemoryInfo"
  _wrap_COrtMemoryInfo ::
    ConstPtr OrtApi ->
    Ptr COrtMemoryInfo ->
    IO (Ptr OrtMemoryInfo)

#{def
  HsOrtMemoryInfo* _wrap_COrtMemoryInfo(
    const OrtApi* ortApi,
    COrtMemoryInfo* ortMemoryInfo
  ) {
    HsOrtMemoryInfo *out = malloc(sizeof *out);
    out->ortApi = ortApi;
    out->ortMemoryInfo = ortMemoryInfo;
    return out;
  }
}

foreign import capi unsafe
  "Onnxruntime/CApi_hsc.h &_wrap_OrtApi_ReleaseMemoryInfo"
  _wrap_OrtApi_ReleaseMemoryInfo ::
    FunPtr (
      Ptr OrtMemoryInfo ->
      IO ()
    )

#{def
  void _wrap_OrtApi_ReleaseMemoryInfo(HsOrtMemoryInfo* ortMemoryInfo) {
    ortMemoryInfo->ortApi->ReleaseMemoryInfo(ortMemoryInfo->ortMemoryInfo);
    free(ortMemoryInfo);
  }
}

-------------------------------------------------------------------------------
-- OrtSession

data
  {-# CTYPE "onnxruntime_c_api.h" "COrtSession" #-}
  COrtSession

#{def
  typedef OrtSession COrtSession;
}

newtype
  {-# CTYPE "Onnxruntime/CApi_hsc.h" "HsOrtSession" #-}
  OrtSession = OrtSession { ortSessionForeignPtr :: ForeignPtr OrtSession }

#{def
  typedef struct HsOrtSession {
    const OrtApi* ortApi;
    COrtSession* ortSession;
  } HsOrtSession;
}

instance HasOrtApi OrtSession where
  type CType OrtSession = COrtSession
  getOrtApi ortSession =
    withOrtSessionPtr ortSession $ \ortSessionPtr ->
      OrtApi <$> #{peek HsOrtSession, ortApi} ortSessionPtr
  withCTypePtr = withCOrtSessionPtr

-- | Internal helper.
withOrtSessionPtr ::
  OrtSession ->
  (Ptr OrtSession -> IO a) ->
  IO a
withOrtSessionPtr ortSession =
  withForeignPtr ortSession.ortSessionForeignPtr

-- | Internal helper.
withCOrtSessionPtr ::
  OrtSession ->
  (Ptr COrtSession -> IO a) ->
  IO a
withCOrtSessionPtr ortSession action =
  withOrtSessionPtr ortSession $ \ortSessionPtr -> do
    cOrtSessionPtr <- #{peek HsOrtSession, ortSession} ortSessionPtr
    action cOrtSessionPtr

-- | Internal helper.
wrapCOrtSession ::
  OrtApi ->
  Ptr COrtSession ->
  IO OrtSession
wrapCOrtSession ortApi rawOrtSessionPtr = do
  ortSessionPtr <- _wrap_COrtSession ortApi.ortApiConstPtr rawOrtSessionPtr
  ortSessionForeignPtr <- newForeignPtr _wrap_OrtApi_ReleaseSession ortSessionPtr
  pure $ OrtSession ortSessionForeignPtr

foreign import capi unsafe
  "Onnxruntime/CApi_hsc.h _wrap_COrtSession"
  _wrap_COrtSession ::
    ConstPtr OrtApi ->
    Ptr COrtSession ->
    IO (Ptr OrtSession)

#{def
  HsOrtSession* _wrap_COrtSession(
    const OrtApi* ortApi,
    COrtSession* ortSession
  ) {
    HsOrtSession *out = malloc(sizeof *out);
    out->ortApi = ortApi;
    out->ortSession = ortSession;
    return out;
  }
}

foreign import capi unsafe
  "Onnxruntime/CApi_hsc.h &_wrap_OrtApi_ReleaseSession"
  _wrap_OrtApi_ReleaseSession ::
    FunPtr (
      Ptr OrtSession ->
      IO ()
    )

#{def
  void _wrap_OrtApi_ReleaseSession(HsOrtSession* ortSession) {
    ortSession->ortApi->ReleaseSession(ortSession->ortSession);
    free(ortSession);
  }
}

-------------------------------------------------------------------------------
-- OrtSessionOptions

data
  {-# CTYPE "onnxruntime_c_api.h" "COrtSessionOptions" #-}
  COrtSessionOptions

#{def
  typedef OrtSessionOptions COrtSessionOptions;
}

newtype
  {-# CTYPE "Onnxruntime/CApi_hsc.h" "HsOrtSessionOptions" #-}
  OrtSessionOptions = OrtSessionOptions { ortSessionOptionsForeignPtr :: ForeignPtr OrtSessionOptions }

#{def
  typedef struct HsOrtSessionOptions {
    const OrtApi* ortApi;
    COrtSessionOptions* ortSessionOptions;
  } HsOrtSessionOptions;
}

instance HasOrtApi OrtSessionOptions where
  type CType OrtSessionOptions = COrtSessionOptions
  getOrtApi ortSessionOptions =
    withOrtSessionOptionsPtr ortSessionOptions $ \ortSessionOptionsPtr ->
      OrtApi <$> #{peek HsOrtSessionOptions, ortApi} ortSessionOptionsPtr
  withCTypePtr = withCOrtSessionOptionsPtr

-- | Internal helper.
withOrtSessionOptionsPtr ::
  OrtSessionOptions ->
  (Ptr OrtSessionOptions -> IO a) ->
  IO a
withOrtSessionOptionsPtr ortSessionOptions =
  withForeignPtr ortSessionOptions.ortSessionOptionsForeignPtr

-- | Internal helper.
withCOrtSessionOptionsPtr ::
  OrtSessionOptions ->
  (Ptr COrtSessionOptions -> IO a) ->
  IO a
withCOrtSessionOptionsPtr ortSessionOptions action =
  withOrtSessionOptionsPtr ortSessionOptions $ \ortSessionOptionsPtr -> do
    cOrtSessionOptionsPtr <- #{peek HsOrtSessionOptions, ortSessionOptions} ortSessionOptionsPtr
    action cOrtSessionOptionsPtr

-- | Internal helper.
wrapCOrtSessionOptions ::
  OrtApi ->
  Ptr COrtSessionOptions ->
  IO OrtSessionOptions
wrapCOrtSessionOptions ortApi rawOrtSessionOptionsPtr = do
  ortSessionOptionsPtr <- _wrap_COrtSessionOptions ortApi.ortApiConstPtr rawOrtSessionOptionsPtr
  ortSessionOptionsForeignPtr <- newForeignPtr _wrap_OrtApi_ReleaseSessionOptions ortSessionOptionsPtr
  pure $ OrtSessionOptions ortSessionOptionsForeignPtr

foreign import capi unsafe
  "Onnxruntime/CApi_hsc.h _wrap_COrtSessionOptions"
  _wrap_COrtSessionOptions ::
    ConstPtr OrtApi ->
    Ptr COrtSessionOptions ->
    IO (Ptr OrtSessionOptions)

#{def
  HsOrtSessionOptions* _wrap_COrtSessionOptions(
    const OrtApi* ortApi,
    COrtSessionOptions* ortSessionOptions
  ) {
    HsOrtSessionOptions *out = malloc(sizeof *out);
    out->ortApi = ortApi;
    out->ortSessionOptions = ortSessionOptions;
    return out;
  }
}

foreign import capi unsafe
  "Onnxruntime/CApi_hsc.h &_wrap_OrtApi_ReleaseSessionOptions"
  _wrap_OrtApi_ReleaseSessionOptions ::
    FunPtr (
      Ptr OrtSessionOptions ->
      IO ()
    )

#{def
  void _wrap_OrtApi_ReleaseSessionOptions(HsOrtSessionOptions* ortSessionOptions) {
    ortSessionOptions->ortApi->ReleaseSessionOptions(ortSessionOptions->ortSessionOptions);
    free(ortSessionOptions);
  }
}

-------------------------------------------------------------------------------
-- OrtStatus

data
  {-# CTYPE "onnxruntime_c_api.h" "OrtStatus" #-}
  OrtStatus

foreign import capi unsafe
  "Onnxruntime/CApi_hsc.h _wrap_OrtApi_ReleaseStatus"
  _wrap_OrtApi_ReleaseStatus ::
    OrtApi ->
    Ptr OrtStatus ->
    IO ()

#{def
  void _wrap_OrtApi_ReleaseStatus(const OrtApi* ortApi, OrtStatus* ortStatus) {
    ortApi->ReleaseStatus(ortStatus);
  }
}

data OrtError = OrtError
  { ortErrorCode    :: {-# UNPACK #-} !OrtErrorCode
  , ortErrorMessage :: {-# UNPACK #-} !ByteString
  }
  deriving stock (Eq, Show)

instance Exception OrtError where
  displayException ortError =
    printf "ERROR[%s]: %s"
      (displayException ortError.ortErrorCode)
      (BSC.unpack ortError.ortErrorMessage)

handleOrtStatus ::
  OrtApi ->
  Ptr OrtStatus ->
  IO a ->
  IO a
handleOrtStatus ortApi ortStatusPtr action
  | ortStatusPtr == nullPtr = action
  | otherwise = do
    let actionOrError = do
          ortErrorCode <- ortApiGetErrorCode ortApi ortStatusPtr
          if ortErrorCode == OrtOk then action else do
            ortErrorMessage <- ortApiGetErrorMessage ortApi ortStatusPtr
            throwIO OrtError {..}
    let cleanupStatus =
          _wrap_OrtApi_ReleaseStatus ortApi ortStatusPtr
    actionOrError `finally` cleanupStatus

-------------------------------------------------------------------------------
-- OrtTensorTypeAndShapeInfo

data
  {-# CTYPE "onnxruntime_c_api.h" "COrtTensorTypeAndShapeInfo" #-}
  COrtTensorTypeAndShapeInfo

#{def
  typedef OrtTensorTypeAndShapeInfo COrtTensorTypeAndShapeInfo;
}

data
  {-# CTYPE "Onnxruntime/CApi_hsc.h" "HsOrtTensorTypeAndShapeInfo" #-}
  OrtTensorTypeAndShapeInfo
    = OrtTensorTypeAndShapeInfo { ortTensorTypeAndShapeInfoForeignPtr :: ForeignPtr OrtTensorTypeAndShapeInfo }
    | OrtTensorTypeAndShapeInfoFromOrtTypeInfo { ortTypeInfo :: OrtTypeInfo, ortTensorTypeAndShapeInfoForeignPtr :: ForeignPtr OrtTensorTypeAndShapeInfo }

#{def
  typedef struct HsOrtTensorTypeAndShapeInfo {
    const OrtApi* ortApi;
    COrtTensorTypeAndShapeInfo* ortTensorTypeAndShapeInfo;
  } HsOrtTensorTypeAndShapeInfo;
}

instance HasOrtApi OrtTensorTypeAndShapeInfo where
  type CType OrtTensorTypeAndShapeInfo = COrtTensorTypeAndShapeInfo
  getOrtApi ortTensorTypeAndShapeInfo =
    withOrtTensorTypeAndShapeInfoPtr ortTensorTypeAndShapeInfo $ \ortTensorTypeAndShapeInfoPtr ->
      OrtApi <$> #{peek HsOrtTensorTypeAndShapeInfo, ortApi} ortTensorTypeAndShapeInfoPtr
  withCTypePtr = withCOrtTensorTypeAndShapeInfoPtr

-- | Internal helper.
withOrtTensorTypeAndShapeInfoPtr ::
  OrtTensorTypeAndShapeInfo ->
  (Ptr OrtTensorTypeAndShapeInfo -> IO a) ->
  IO a
withOrtTensorTypeAndShapeInfoPtr ortTensorTypeAndShapeInfo =
  withForeignPtr ortTensorTypeAndShapeInfo.ortTensorTypeAndShapeInfoForeignPtr

-- | Internal helper.
withCOrtTensorTypeAndShapeInfoPtr ::
  OrtTensorTypeAndShapeInfo ->
  (Ptr COrtTensorTypeAndShapeInfo -> IO a) ->
  IO a
withCOrtTensorTypeAndShapeInfoPtr ortTensorTypeAndShapeInfo action =
  withOrtTensorTypeAndShapeInfoPtr ortTensorTypeAndShapeInfo $ \ortTensorTypeAndShapeInfoPtr -> do
    cOrtTensorTypeAndShapeInfoPtr <- #{peek HsOrtTensorTypeAndShapeInfo, ortTensorTypeAndShapeInfo} ortTensorTypeAndShapeInfoPtr
    action cOrtTensorTypeAndShapeInfoPtr

-- | Internal helper.
wrapCOrtTensorTypeAndShapeInfo ::
  OrtApi ->
  Ptr COrtTensorTypeAndShapeInfo ->
  IO OrtTensorTypeAndShapeInfo
wrapCOrtTensorTypeAndShapeInfo ortApi rawOrtTensorTypeAndShapeInfoPtr = do
  ortTensorTypeAndShapeInfoPtr <- _wrap_COrtTensorTypeAndShapeInfo ortApi.ortApiConstPtr rawOrtTensorTypeAndShapeInfoPtr
  ortTensorTypeAndShapeInfoForeignPtr <- newForeignPtr _wrap_OrtApi_ReleaseTensorTypeAndShapeInfo ortTensorTypeAndShapeInfoPtr
  pure $ OrtTensorTypeAndShapeInfo ortTensorTypeAndShapeInfoForeignPtr

foreign import capi unsafe
  "Onnxruntime/CApi_hsc.h _wrap_COrtTensorTypeAndShapeInfo"
  _wrap_COrtTensorTypeAndShapeInfo ::
    ConstPtr OrtApi ->
    Ptr COrtTensorTypeAndShapeInfo ->
    IO (Ptr OrtTensorTypeAndShapeInfo)

#{def
  HsOrtTensorTypeAndShapeInfo* _wrap_COrtTensorTypeAndShapeInfo(
    const OrtApi* ortApi,
    COrtTensorTypeAndShapeInfo* ortTensorTypeAndShapeInfo
  ) {
    HsOrtTensorTypeAndShapeInfo *out = malloc(sizeof *out);
    out->ortApi = ortApi;
    out->ortTensorTypeAndShapeInfo = ortTensorTypeAndShapeInfo;
    return out;
  }
}

foreign import capi unsafe
  "Onnxruntime/CApi_hsc.h &_wrap_OrtApi_ReleaseTensorTypeAndShapeInfo"
  _wrap_OrtApi_ReleaseTensorTypeAndShapeInfo ::
    FunPtr (
      Ptr OrtTensorTypeAndShapeInfo ->
      IO ()
    )

#{def
  void _wrap_OrtApi_ReleaseTensorTypeAndShapeInfo(HsOrtTensorTypeAndShapeInfo* ortTensorTypeAndShapeInfo) {
    ortTensorTypeAndShapeInfo->ortApi->ReleaseTensorTypeAndShapeInfo(ortTensorTypeAndShapeInfo->ortTensorTypeAndShapeInfo);
    free(ortTensorTypeAndShapeInfo);
  }
}

-- | Internal helper.
wrapCOrtTensorTypeAndShapeInfoFromOrtTypeInfo ::
  OrtApi ->
  OrtTypeInfo ->
  Ptr COrtTensorTypeAndShapeInfo ->
  IO OrtTensorTypeAndShapeInfo
wrapCOrtTensorTypeAndShapeInfoFromOrtTypeInfo ortApi ortTypeInfo rawOrtTensorTypeAndShapeInfoPtr = do
  ortTensorTypeAndShapeInfoPtr <- _wrap_COrtTensorTypeAndShapeInfo ortApi.ortApiConstPtr rawOrtTensorTypeAndShapeInfoPtr
  ortTensorTypeAndShapeInfoForeignPtr <- newForeignPtr _wrap_OrtApi_OrtTensorTypeAndShapeInfoFromOrtTypeInfo ortTensorTypeAndShapeInfoPtr
  pure $ OrtTensorTypeAndShapeInfoFromOrtTypeInfo ortTypeInfo ortTensorTypeAndShapeInfoForeignPtr

foreign import capi unsafe
  "Onnxruntime/CApi_hsc.h &_wrap_OrtApi_OrtTensorTypeAndShapeInfoFromOrtTypeInfo"
  _wrap_OrtApi_OrtTensorTypeAndShapeInfoFromOrtTypeInfo ::
    FunPtr (
      Ptr OrtTensorTypeAndShapeInfo ->
      IO ()
    )

#{def
  void _wrap_OrtApi_OrtTensorTypeAndShapeInfoFromOrtTypeInfo(HsOrtTensorTypeAndShapeInfo* ortTensorTypeAndShapeInfo) {
    free(ortTensorTypeAndShapeInfo);
  }
}

-------------------------------------------------------------------------------
-- OrtTypeInfo

data
  {-# CTYPE "onnxruntime_c_api.h" "COrtTypeInfo" #-}
  COrtTypeInfo

#{def
  typedef OrtTypeInfo COrtTypeInfo;
}

newtype
  {-# CTYPE "Onnxruntime/CApi_hsc.h" "HsOrtTypeInfo" #-}
  OrtTypeInfo = OrtTypeInfo { ortTypeInfoForeignPtr :: ForeignPtr OrtTypeInfo }

#{def
  typedef struct HsOrtTypeInfo {
    const OrtApi* ortApi;
    COrtTypeInfo* ortTypeInfo;
  } HsOrtTypeInfo;
}

instance HasOrtApi OrtTypeInfo where
  type CType OrtTypeInfo = COrtTypeInfo
  getOrtApi ortTypeInfo =
    withOrtTypeInfoPtr ortTypeInfo $ \ortTypeInfoPtr ->
      OrtApi <$> #{peek HsOrtTypeInfo, ortApi} ortTypeInfoPtr
  withCTypePtr = withCOrtTypeInfoPtr

-- | Internal helper.
withOrtTypeInfoPtr ::
  OrtTypeInfo ->
  (Ptr OrtTypeInfo -> IO a) ->
  IO a
withOrtTypeInfoPtr ortTypeInfo =
  withForeignPtr ortTypeInfo.ortTypeInfoForeignPtr

-- | Internal helper.
withCOrtTypeInfoPtr ::
  OrtTypeInfo ->
  (Ptr COrtTypeInfo -> IO a) ->
  IO a
withCOrtTypeInfoPtr ortTypeInfo action =
  withOrtTypeInfoPtr ortTypeInfo $ \ortTypeInfoPtr -> do
    cOrtTypeInfoPtr <- #{peek HsOrtTypeInfo, ortTypeInfo} ortTypeInfoPtr
    action cOrtTypeInfoPtr

-- | Internal helper.
wrapCOrtTypeInfo ::
  OrtApi ->
  Ptr COrtTypeInfo ->
  IO OrtTypeInfo
wrapCOrtTypeInfo ortApi rawOrtTypeInfoPtr = do
  ortTypeInfoPtr <- _wrap_COrtTypeInfo ortApi.ortApiConstPtr rawOrtTypeInfoPtr
  ortTypeInfoForeignPtr <- newForeignPtr _wrap_OrtApi_ReleaseTypeInfo ortTypeInfoPtr
  pure $ OrtTypeInfo ortTypeInfoForeignPtr

foreign import capi unsafe
  "Onnxruntime/CApi_hsc.h _wrap_COrtTypeInfo"
  _wrap_COrtTypeInfo ::
    ConstPtr OrtApi ->
    Ptr COrtTypeInfo ->
    IO (Ptr OrtTypeInfo)

#{def
  HsOrtTypeInfo* _wrap_COrtTypeInfo(
    const OrtApi* ortApi,
    COrtTypeInfo* ortTypeInfo
  ) {
    HsOrtTypeInfo *out = malloc(sizeof *out);
    out->ortApi = ortApi;
    out->ortTypeInfo = ortTypeInfo;
    return out;
  }
}

foreign import capi unsafe
  "Onnxruntime/CApi_hsc.h &_wrap_OrtApi_ReleaseTypeInfo"
  _wrap_OrtApi_ReleaseTypeInfo ::
    FunPtr (
      Ptr OrtTypeInfo ->
      IO ()
    )

#{def
  void _wrap_OrtApi_ReleaseTypeInfo(HsOrtTypeInfo* ortTypeInfo) {
    ortTypeInfo->ortApi->ReleaseTypeInfo(ortTypeInfo->ortTypeInfo);
    free(ortTypeInfo);
  }
}

-------------------------------------------------------------------------------
-- OrtRunOptions

data
  {-# CTYPE "onnxruntime_c_api.h" "COrtRunOptions" #-}
  COrtRunOptions

#{def
  typedef OrtRunOptions COrtRunOptions;
}

newtype
  {-# CTYPE "Onnxruntime/CApi_hsc.h" "HsOrtRunOptions" #-}
  OrtRunOptions = OrtRunOptions { ortRunOptionsForeignPtr :: ForeignPtr OrtRunOptions }

#{def
  typedef struct HsOrtRunOptions {
    const OrtApi* ortApi;
    COrtRunOptions* ortRunOptions;
  } HsOrtRunOptions;
}

instance HasOrtApi OrtRunOptions where
  type CType OrtRunOptions = COrtRunOptions
  getOrtApi ortRunOptions =
    withOrtRunOptionsPtr ortRunOptions $ \ortRunOptionsPtr ->
      OrtApi <$> #{peek HsOrtRunOptions, ortApi} ortRunOptionsPtr
  withCTypePtr = withCOrtRunOptionsPtr

-- | Internal helper.
withOrtRunOptionsPtr ::
  OrtRunOptions ->
  (Ptr OrtRunOptions -> IO a) ->
  IO a
withOrtRunOptionsPtr ortRunOptions =
  withForeignPtr ortRunOptions.ortRunOptionsForeignPtr

-- | Internal helper.
withCOrtRunOptionsPtr ::
  OrtRunOptions ->
  (Ptr COrtRunOptions -> IO a) ->
  IO a
withCOrtRunOptionsPtr ortRunOptions action =
  withOrtRunOptionsPtr ortRunOptions $ \ortRunOptionsPtr -> do
    cOrtRunOptionsPtr <- #{peek HsOrtRunOptions, ortRunOptions} ortRunOptionsPtr
    action cOrtRunOptionsPtr

-- | Internal helper.
wrapCOrtRunOptions ::
  OrtApi ->
  Ptr COrtRunOptions ->
  IO OrtRunOptions
wrapCOrtRunOptions ortApi rawOrtRunOptionsPtr = do
  ortRunOptionsPtr <- _wrap_COrtRunOptions ortApi.ortApiConstPtr rawOrtRunOptionsPtr
  ortRunOptionsForeignPtr <- newForeignPtr _wrap_OrtApi_ReleaseRunOptions ortRunOptionsPtr
  pure $ OrtRunOptions ortRunOptionsForeignPtr

foreign import capi unsafe
  "Onnxruntime/CApi_hsc.h _wrap_COrtRunOptions"
  _wrap_COrtRunOptions ::
    ConstPtr OrtApi ->
    Ptr COrtRunOptions ->
    IO (Ptr OrtRunOptions)

#{def
  HsOrtRunOptions* _wrap_COrtRunOptions(
    const OrtApi* ortApi,
    COrtRunOptions* ortRunOptions
  ) {
    HsOrtRunOptions *out = malloc(sizeof *out);
    out->ortApi = ortApi;
    out->ortRunOptions = ortRunOptions;
    return out;
  }
}

foreign import capi unsafe
  "Onnxruntime/CApi_hsc.h &_wrap_OrtApi_ReleaseRunOptions"
  _wrap_OrtApi_ReleaseRunOptions ::
    FunPtr (
      Ptr OrtRunOptions ->
      IO ()
    )

#{def
  void _wrap_OrtApi_ReleaseRunOptions(HsOrtRunOptions* ortRunOptions) {
    ortRunOptions->ortApi->ReleaseRunOptions(ortRunOptions->ortRunOptions);
    free(ortRunOptions);
  }
}

-------------------------------------------------------------------------------
-- OrtValue

data
  {-# CTYPE "onnxruntime_c_api.h" "COrtValue" #-}
  COrtValue

#{def
  typedef OrtValue COrtValue;
}

newtype
  {-# CTYPE "Onnxruntime/CApi_hsc.h" "HsOrtValue" #-}
  OrtValue = OrtValue { ortValueForeignPtr :: ForeignPtr OrtValue }

#{def
  typedef struct HsOrtValue {
    const OrtApi* ortApi;
    COrtValue* ortValue;
  } HsOrtValue;
}

instance HasOrtApi OrtValue where
  type CType OrtValue = COrtValue
  getOrtApi ortValue =
    withOrtValuePtr ortValue $ \ortValuePtr ->
      OrtApi <$> #{peek HsOrtValue, ortApi} ortValuePtr
  withCTypePtr = withCOrtValuePtr

-- | Internal helper.
withOrtValuePtr ::
  OrtValue ->
  (Ptr OrtValue -> IO a) ->
  IO a
withOrtValuePtr ortValue =
  withForeignPtr ortValue.ortValueForeignPtr

-- | Internal helper.
withCOrtValuePtr ::
  OrtValue ->
  (Ptr COrtValue -> IO a) ->
  IO a
withCOrtValuePtr ortValue action =
  withOrtValuePtr ortValue $ \ortValuePtr -> do
    cOrtValuePtr <- #{peek HsOrtValue, ortValue} ortValuePtr
    action cOrtValuePtr

-- | Internal helper.
wrapCOrtValue ::
  OrtApi ->
  Ptr COrtValue ->
  IO OrtValue
wrapCOrtValue ortApi rawOrtValuePtr = do
  ortValuePtr <- _wrap_COrtValue ortApi.ortApiConstPtr rawOrtValuePtr
  ortValueForeignPtr <- newForeignPtr _wrap_OrtApi_ReleaseValue ortValuePtr
  pure $ OrtValue ortValueForeignPtr

foreign import capi unsafe
  "Onnxruntime/CApi_hsc.h _wrap_COrtValue"
  _wrap_COrtValue ::
    ConstPtr OrtApi ->
    Ptr COrtValue ->
    IO (Ptr OrtValue)

#{def
  HsOrtValue* _wrap_COrtValue(
    const OrtApi* ortApi,
    COrtValue* ortValue
  ) {
    HsOrtValue *out = malloc(sizeof *out);
    out->ortApi = ortApi;
    out->ortValue = ortValue;
    return out;
  }
}

foreign import capi unsafe
  "Onnxruntime/CApi_hsc.h &_wrap_OrtApi_ReleaseValue"
  _wrap_OrtApi_ReleaseValue ::
    FunPtr (
      Ptr OrtValue ->
      IO ()
    )

#{def
  void _wrap_OrtApi_ReleaseValue(HsOrtValue* ortValue) {
    ortValue->ortApi->ReleaseValue(ortValue->ortValue);
    free(ortValue);
  }
}

-------------------------------------------------------------------------------
-- ONNX Runtime: API Function
-------------------------------------------------------------------------------

-------------------------------------------------------------------------------
-- OrtApi::GetErrorCode

foreign import capi unsafe
  "Onnxruntime/CApi_hsc.h _wrap_OrtApi_GetErrorCode"
  ortApiGetErrorCode ::
    OrtApi ->
    Ptr OrtStatus ->
    IO OrtErrorCode

#{def
  OrtErrorCode _wrap_OrtApi_GetErrorCode(const OrtApi* ortApi, OrtStatus* ortStatus) {
    return ortApi->GetErrorCode(ortStatus);
  }
}

-------------------------------------------------------------------------------
-- OrtApi::GetErrorMessage

ortApiGetErrorMessageAsString ::
  OrtApi ->
  Ptr OrtStatus ->
  IO String
ortApiGetErrorMessageAsString ortApi ortStatusPtr = do
  ConstPtr msgPtr <- _wrap_OrtApi_GetErrorMessage ortApi ortStatusPtr
  peekCString msgPtr

ortApiGetErrorMessage ::
  OrtApi ->
  Ptr OrtStatus ->
  IO ByteString
ortApiGetErrorMessage ortApi ortStatusPtr = do
  ConstPtr msgPtr <- _wrap_OrtApi_GetErrorMessage ortApi ortStatusPtr
  print (msgPtr == nullPtr)
  BS.packCString msgPtr

foreign import capi unsafe
  "Onnxruntime/CApi_hsc.h _wrap_OrtApi_GetErrorMessage"
  _wrap_OrtApi_GetErrorMessage ::
    OrtApi ->
    Ptr OrtStatus ->
    IO (ConstPtr CChar)

#{def
  const char* _wrap_OrtApi_GetErrorMessage(const OrtApi* ortApi, OrtStatus* ortStatus) {
    return ortApi->GetErrorMessage(ortStatus);
  }
}

-------------------------------------------------------------------------------
-- OrtApi::CreateEnv

{- |
> ORT_API2_STATUS(CreateEnv,
>   OrtLoggingLevel log_severity_level,
>   _In_ const char* logid,
>   _Outptr_ OrtEnv** out
> );
-}
ortApiCreateEnv ::
  OrtApi ->
  OrtLoggingLevel ->
  String ->
  IO OrtEnv
ortApiCreateEnv ortApi logSeverityLevel logid = do
  withCString logid $ \logidPtr -> do
    alloca $ \outPtr -> do
      ortStatusPtr <-
        _wrap_OrtApi_CreateEnv
          ortApi.ortApiConstPtr
          logSeverityLevel
          (ConstPtr logidPtr) -- NOTE: This is unsafe.
          outPtr
      handleOrtStatus ortApi ortStatusPtr $ do
        wrapCOrtEnv ortApi
          =<< peek outPtr

foreign import capi unsafe
  "Onnxruntime/CApi_hsc.h _wrap_OrtApi_CreateEnv"
  _wrap_OrtApi_CreateEnv ::
    ConstPtr OrtApi ->
    OrtLoggingLevel ->
    ConstPtr CChar ->
    Ptr (Ptr COrtEnv) ->
    IO (Ptr OrtStatus)

#{def
  OrtStatus* _wrap_OrtApi_CreateEnv(
    const OrtApi* ortApi,
    OrtLoggingLevel logSeverityLevel,
    const char* logid,
    COrtEnv** out
  ) {
    return ortApi->CreateEnv(logSeverityLevel, logid, out);
  }
}

-------------------------------------------------------------------------------
-- OrtApi::CreateEnvWithCustomLogger

{-
> ORT_API2_STATUS(CreateEnvWithCustomLogger,
>   _In_ OrtLoggingFunction logging_function,
>   _In_opt_ void* logger_param,
>   _In_ OrtLoggingLevel log_severity_level,
>   _In_ const char* logid,
>   _Outptr_ OrtEnv** out
> );
-}

-- TODO: unimplemented

-------------------------------------------------------------------------------
-- OrtApi::EnableTelemetryEvents

{-
> ORT_API2_STATUS(EnableTelemetryEvents,
>   _In_ const OrtEnv* env
>);
-}

-- TODO: unimplemented

-------------------------------------------------------------------------------
-- OrtApi::DisableTelemetryEvents

{-
> ORT_API2_STATUS(DisableTelemetryEvents,
>   _In_ const OrtEnv* env
>);
-}

-- TODO: unimplemented

-------------------------------------------------------------------------------
-- OrtApi::CreateSession

{-
> ORT_API2_STATUS(CreateSession,
>   _In_ const OrtEnv* env,
>   _In_ const ORTCHAR_T* model_path,
>   _In_ const OrtSessionOptions* options,
>   _Outptr_ OrtSession** out
> );
-}
ortApiCreateSession ::
  OrtEnv ->
  FilePath ->
  OrtSessionOptions ->
  IO OrtSession
ortApiCreateSession ortEnv modelPath options = do
  ortApi <- getOrtApi ortEnv
  alloca $ \outPtr -> do
    withCTypePtr ortEnv $ \cOrtEnvPtr -> do
      withCString modelPath $ \modelPathPtr -> do
        withCTypePtr options $ \cOrtSessionOptionsPtr -> do
          ortStatusPtr <-
            _wrap_OrtApi_CreateSession
              ortApi
              cOrtEnvPtr
              (ConstPtr modelPathPtr) -- NOTE: This is unsafe.
              cOrtSessionOptionsPtr
              outPtr
          handleOrtStatus ortApi ortStatusPtr $ do
            wrapCOrtSession ortApi
              =<< peek outPtr

foreign import capi unsafe
  "Onnxruntime/CApi_hsc.h _wrap_OrtApi_CreateSession"
  _wrap_OrtApi_CreateSession ::
    OrtApi ->
    Ptr COrtEnv ->
    ConstPtr CChar ->
    Ptr COrtSessionOptions ->
    Ptr (Ptr COrtSession) ->
    IO (Ptr OrtStatus)

#{def
  OrtStatus* _wrap_OrtApi_CreateSession(
    OrtApi* ortApi,
    COrtEnv* ortEnv,
    const ORTCHAR_T* modelPath,
    COrtSessionOptions* options,
    COrtSession** out
  ) {
    return ortApi->CreateSession(
      ortEnv,
      modelPath,
      options,
      out
    );
  }
}

-------------------------------------------------------------------------------
-- OrtApi::CreateSessionFromArray

{-
> ORT_API2_STATUS(CreateSessionFromArray,
>   _In_ const OrtEnv* env,
>   _In_ const void* model_data,
>   size_t model_data_length,
>   _In_ const OrtSessionOptions* options,
>   _Outptr_ OrtSession** out
> );
-}

-- TODO: unimplemented

-------------------------------------------------------------------------------
-- OrtApi::Run

{-
> ORT_API2_STATUS(Run,
>   _Inout_ OrtSession* session,
>   _In_opt_ const OrtRunOptions* run_options,
>   _In_reads_(input_len) const char* const* input_names,
>   _In_reads_(input_len) const OrtValue* const* inputs,
>   size_t input_len,
>   _In_reads_(output_names_len) const char* const* output_names,
>   size_t output_names_len,
>   _Inout_updates_all_(output_names_len) OrtValue** outputs
> );
-}
ortApiRun ::
  OrtSession ->
  OrtRunOptions ->
  [String] ->
  [OrtValue] ->
  [String] ->
  IO [OrtValue]
ortApiRun ortSession ortRunOptions inputNames inputs outputNames = do
  ortApi <- getOrtApi ortSession
  alloca @(Ptr COrtValue) $ \outputsPtr ->
    withCTypePtr ortSession $ \cOrtSessionPtr ->
      withCTypePtr ortRunOptions $ \cOrtRunOptionsPtr ->
        withCStringArrayLen inputNames $ \inputLen cInputNames ->
          withCTypeArrayLen inputs $ \inputLen' cInputs ->
            withCStringArrayLen outputNames $ \outputLen cOutputNames ->
              -- TODO turn into a proper exception
              assert (inputLen == inputLen') $ do
                ortStatusPtr <-
                  _wrap_OrtApi_Run
                    ortApi
                    cOrtSessionPtr
                    cOrtRunOptionsPtr
                    (coerce cInputNames)
                    (coerce cInputs)
                    (fromIntegral inputLen)
                    (coerce cOutputNames)
                    (fromIntegral outputLen)
                    outputsPtr

                handleOrtStatus ortApi ortStatusPtr $ do
                  traverse (wrapCOrtValue ortApi)
                    =<< peekArray outputLen outputsPtr

foreign import capi unsafe
  "Onnxruntime/CApi_hsc.h _wrap_OrtApi_Run"
  _wrap_OrtApi_Run ::
    OrtApi ->
    Ptr COrtSession ->
    Ptr COrtRunOptions ->
    ConstPtr (ConstPtr CChar) ->
    ConstPtr (ConstPtr COrtValue) ->
    ( #{type size_t} ) ->
    ConstPtr (ConstPtr CChar) ->
    ( #{type size_t} ) ->
    Ptr (Ptr COrtValue) ->
    IO (Ptr OrtStatus)

#{def
  OrtStatus* _wrap_OrtApi_Run(
    OrtApi* ortApi,
    COrtSession* session,
    COrtRunOptions* run_options,
    const char* const* input_names,
    const COrtValue* const* inputs,
    size_t input_len,
    const char* const* output_names,
    size_t output_names_len,
    COrtValue** outputs
  ) {
    return ortApi->Run(
      session,
      run_options,
      input_names,
      inputs,
      input_len,
      output_names,
      output_names_len,
      outputs
    );
  }
}
-------------------------------------------------------------------------------
-- OrtApi::CreateSessionOptions

{- |
> ORT_API2_STATUS(CreateSessionOptions,
>   _Outptr_ OrtSessionOptions** options
> );
-}
ortApiCreateSessionOptions ::
  OrtApi ->
  IO OrtSessionOptions
ortApiCreateSessionOptions ortApi = do
  alloca $ \outPtr -> do
    ortStatusPtr <-
      _wrap_OrtApi_CreateSessionOptions
        ortApi.ortApiConstPtr
        outPtr
    handleOrtStatus ortApi ortStatusPtr $ do
      cOrtSessionOptionsPtr <- peek outPtr
      wrapCOrtSessionOptions ortApi cOrtSessionOptionsPtr

foreign import capi unsafe
  "Onnxruntime/CApi_hsc.h _wrap_OrtApi_CreateSessionOptions"
  _wrap_OrtApi_CreateSessionOptions ::
    ConstPtr OrtApi ->
    Ptr (Ptr COrtSessionOptions) ->
    IO (Ptr OrtStatus)

#{def
  OrtStatus* _wrap_OrtApi_CreateSessionOptions(
    const OrtApi* ortApi,
    COrtSessionOptions** out
  ) {
    return ortApi->CreateSessionOptions(out);
  }
}

-------------------------------------------------------------------------------
-- OrtApi::CloneSessionOptions

{- |
> ORT_API2_STATUS(CloneSessionOptions,
>   _In_ const OrtSessionOptions* in_options,
>   _Outptr_ OrtSessionOptions** out_options
  );
-}
ortApiCloneSessionOptions ::
  OrtSessionOptions ->
  IO OrtSessionOptions
ortApiCloneSessionOptions inOptions = do
  ortApi <- getOrtApi inOptions
  withOrtSessionOptionsPtr inOptions $ \inOptionsPtr -> do
    alloca $ \outPtr -> do
      ortStatusPtr <-
        _wrap_OrtApi_CloneSessionOptions
          inOptionsPtr
          outPtr
      handleOrtStatus ortApi ortStatusPtr $
        wrapCOrtSessionOptions ortApi
          =<< peek outPtr

foreign import capi unsafe
  "Onnxruntime/CApi_hsc.h _wrap_OrtApi_CloneSessionOptions"
  _wrap_OrtApi_CloneSessionOptions ::
    Ptr OrtSessionOptions ->
    Ptr (Ptr COrtSessionOptions) ->
    IO (Ptr OrtStatus)

#{def
  OrtStatus* _wrap_OrtApi_CloneSessionOptions(
    HsOrtSessionOptions* inOptions,
    COrtSessionOptions** out
  ) {
    return inOptions->ortApi->CloneSessionOptions(
      inOptions->ortSessionOptions,
      out
    );
  }
}

-------------------------------------------------------------------------------
-- OrtApi::SetOptimizedModelFilePath

{- |
> ORT_API2_STATUS(SetOptimizedModelFilePath,
>   _Inout_ OrtSessionOptions* options,
>   _In_ const ORTCHAR_T* optimized_model_filepath
> );
-}
ortApiSetOptimizedModelFilePath ::
  OrtSessionOptions ->
  FilePath ->
  IO ()
ortApiSetOptimizedModelFilePath options optimizedModelFilepath = do
  ortApi <- getOrtApi options
  withCString optimizedModelFilepath $ \optimizedModelFilepathPtr ->
    withOrtSessionOptionsPtr options $ \optionsPtr -> do
      ortStatusPtr <-
        _wrap_OrtApi_SetOptimizedModelFilePath
          optionsPtr
          (ConstPtr optimizedModelFilepathPtr) -- NOTE: This is unsafe.
      handleOrtStatus ortApi ortStatusPtr $ do
        pure ()

foreign import capi unsafe
  "Onnxruntime/CApi_hsc.h _wrap_OrtApi_SetOptimizedModelFilePath"
  _wrap_OrtApi_SetOptimizedModelFilePath ::
    Ptr OrtSessionOptions ->
    ConstPtr CChar ->
    IO (Ptr OrtStatus)

#{def
  OrtStatus* _wrap_OrtApi_SetOptimizedModelFilePath(
    HsOrtSessionOptions* options,
    const ORTCHAR_T* optimizedModelFilepath
  ) {
    return options->ortApi->SetOptimizedModelFilePath(
      options->ortSessionOptions,
      optimizedModelFilepath
    );
  }
}

-------------------------------------------------------------------------------
-- OrtApi::SetSessionExecutionMode

{-
> ORT_API2_STATUS(SetSessionExecutionMode,
>   _Inout_ OrtSessionOptions* options,
>   ExecutionMode execution_mode
> );
-}
ortApiSetSessionExecutionMode ::
  OrtSessionOptions ->
  ExecutionMode ->
  IO ()
ortApiSetSessionExecutionMode options executionMode = do
  ortApi <- getOrtApi options
  withOrtSessionOptionsPtr options $ \optionsPtr -> do
    ortStatusPtr <-
      _wrap_OrtApi_SetSessionExecutionMode
        optionsPtr
        executionMode
    handleOrtStatus ortApi ortStatusPtr $ do
      pure ()

foreign import capi unsafe
  "Onnxruntime/CApi_hsc.h _wrap_OrtApi_SetSessionExecutionMode"
  _wrap_OrtApi_SetSessionExecutionMode ::
    Ptr OrtSessionOptions ->
    ExecutionMode ->
    IO (Ptr OrtStatus)

#{def
  OrtStatus* _wrap_OrtApi_SetSessionExecutionMode(
    HsOrtSessionOptions* options,
    int executionMode
  ) {
    return options->ortApi->SetSessionExecutionMode(
      options->ortSessionOptions,
      executionMode
    );
  }
}

-------------------------------------------------------------------------------
-- OrtApi::EnableProfiling

{- |
> ORT_API2_STATUS(EnableProfiling,
>   _Inout_ OrtSessionOptions* options,
>   _In_ const ORTCHAR_T* profile_file_prefix
> );
-}
ortApiEnableProfiling ::
  OrtSessionOptions ->
  FilePath ->
  IO ()
ortApiEnableProfiling options profileFilePrefix = do
  ortApi <- getOrtApi options
  withCString profileFilePrefix $ \profileFilePrefixPtr ->
    withOrtSessionOptionsPtr options $ \optionsPtr -> do
      ortStatusPtr <-
        _wrap_OrtApi_EnableProfiling
          optionsPtr
          (ConstPtr profileFilePrefixPtr) -- NOTE: This is unsafe.
      handleOrtStatus ortApi ortStatusPtr $ do
        pure ()

foreign import capi unsafe
  "Onnxruntime/CApi_hsc.h _wrap_OrtApi_EnableProfiling"
  _wrap_OrtApi_EnableProfiling ::
    Ptr OrtSessionOptions ->
    ConstPtr CChar ->
    IO (Ptr OrtStatus)

#{def
  OrtStatus* _wrap_OrtApi_EnableProfiling(
    HsOrtSessionOptions* options,
    const ORTCHAR_T* profileFilePrefix
  ) {
    return options->ortApi->EnableProfiling(
      options->ortSessionOptions,
      profileFilePrefix
    );
  }
}

-------------------------------------------------------------------------------
-- OrtApi::DisableProfiling

{- |
> ORT_API2_STATUS(DisableProfiling,
>   _Inout_ OrtSessionOptions* options
> );
-}
ortApiDisableProfiling ::
  OrtSessionOptions ->
  IO ()
ortApiDisableProfiling options = do
  ortApi <- getOrtApi options
  withOrtSessionOptionsPtr options $ \optionsPtr -> do
    ortStatusPtr <-
      _wrap_OrtApi_DisableProfiling
        optionsPtr
    handleOrtStatus ortApi ortStatusPtr $ do
      pure ()

foreign import capi unsafe
  "Onnxruntime/CApi_hsc.h _wrap_OrtApi_DisableProfiling"
  _wrap_OrtApi_DisableProfiling ::
    Ptr OrtSessionOptions ->
    IO (Ptr OrtStatus)

#{def
  OrtStatus* _wrap_OrtApi_DisableProfiling(
    HsOrtSessionOptions* options
  ) {
    return options->ortApi->DisableProfiling(options->ortSessionOptions);
  }
}

-------------------------------------------------------------------------------
-- OrtApi::EnableMemPattern

{- |
> ORT_API2_STATUS(EnableMemPattern,
>   _Inout_ OrtSessionOptions* options
> );
-}
ortApiEnableMemPattern ::
  OrtSessionOptions ->
  IO ()
ortApiEnableMemPattern options = do
  ortApi <- getOrtApi options
  withOrtSessionOptionsPtr options $ \optionsPtr -> do
    ortStatusPtr <-
      _wrap_OrtApi_EnableMemPattern
        optionsPtr
    handleOrtStatus ortApi ortStatusPtr $ do
      pure ()

foreign import capi unsafe
  "Onnxruntime/CApi_hsc.h _wrap_OrtApi_EnableMemPattern"
  _wrap_OrtApi_EnableMemPattern ::
    Ptr OrtSessionOptions ->
    IO (Ptr OrtStatus)

#{def
  OrtStatus* _wrap_OrtApi_EnableMemPattern(
    HsOrtSessionOptions* options
  ) {
    return options->ortApi->EnableMemPattern(options->ortSessionOptions);
  }
}

-------------------------------------------------------------------------------
-- OrtApi::DisableMemPattern

{- |
> ORT_API2_STATUS(DisableMemPattern,
>   _Inout_ OrtSessionOptions* options
> );
-}
ortApiDisableMemPattern ::
  OrtSessionOptions ->
  IO ()
ortApiDisableMemPattern options = do
  ortApi <- getOrtApi options
  withOrtSessionOptionsPtr options $ \optionsPtr -> do
    ortStatusPtr <-
      _wrap_OrtApi_DisableMemPattern
        optionsPtr
    handleOrtStatus ortApi ortStatusPtr $ do
      pure ()

foreign import capi unsafe
  "Onnxruntime/CApi_hsc.h _wrap_OrtApi_DisableMemPattern"
  _wrap_OrtApi_DisableMemPattern ::
    Ptr OrtSessionOptions ->
    IO (Ptr OrtStatus)

#{def
  OrtStatus* _wrap_OrtApi_DisableMemPattern(
    HsOrtSessionOptions* options
  ) {
    return options->ortApi->DisableMemPattern(options->ortSessionOptions);
  }
}

-------------------------------------------------------------------------------
-- OrtApi::EnableCpuMemArena

{- |
> ORT_API2_STATUS(EnableCpuMemArena,
>   _Inout_ OrtSessionOptions* options
> );
-}
ortApiEnableCpuMemArena ::
  OrtSessionOptions ->
  IO ()
ortApiEnableCpuMemArena options = do
  ortApi <- getOrtApi options
  withOrtSessionOptionsPtr options $ \optionsPtr -> do
    ortStatusPtr <-
      _wrap_OrtApi_EnableCpuMemArena
        optionsPtr
    handleOrtStatus ortApi ortStatusPtr $ do
      pure ()

foreign import capi unsafe
  "Onnxruntime/CApi_hsc.h _wrap_OrtApi_EnableCpuMemArena"
  _wrap_OrtApi_EnableCpuMemArena ::
    Ptr OrtSessionOptions ->
    IO (Ptr OrtStatus)

#{def
  OrtStatus* _wrap_OrtApi_EnableCpuMemArena(
    HsOrtSessionOptions* options
  ) {
    return options->ortApi->EnableCpuMemArena(options->ortSessionOptions);
  }
}

-------------------------------------------------------------------------------
-- OrtApi::DisableCpuMemArena

{- |
> ORT_API2_STATUS(DisableCpuMemArena,
>   _Inout_ OrtSessionOptions* options
> );
-}
ortApiDisableCpuMemArena ::
  OrtSessionOptions ->
  IO ()
ortApiDisableCpuMemArena options = do
  ortApi <- getOrtApi options
  withOrtSessionOptionsPtr options $ \optionsPtr -> do
    ortStatusPtr <-
      _wrap_OrtApi_DisableCpuMemArena
        optionsPtr
    handleOrtStatus ortApi ortStatusPtr $ do
      pure ()

foreign import capi unsafe
  "Onnxruntime/CApi_hsc.h _wrap_OrtApi_DisableCpuMemArena"
  _wrap_OrtApi_DisableCpuMemArena ::
    Ptr OrtSessionOptions ->
    IO (Ptr OrtStatus)

#{def
  OrtStatus* _wrap_OrtApi_DisableCpuMemArena(
    HsOrtSessionOptions* options
  ) {
    return options->ortApi->DisableCpuMemArena(options->ortSessionOptions);
  }
}

-------------------------------------------------------------------------------
-- OrtApi::SetSessionLogId

{- |
> ORT_API2_STATUS(SetSessionLogId,
>   _Inout_ OrtSessionOptions* options,
>   const char* logid
> );
-}
ortApiSetSessionLogId ::
  OrtSessionOptions ->
  String ->
  IO ()
ortApiSetSessionLogId options logid = do
  ortApi <- getOrtApi options
  withCString logid $ \logidPtr ->
    withOrtSessionOptionsPtr options $ \optionsPtr -> do
      ortStatusPtr <-
        _wrap_OrtApi_SetSessionLogId
          optionsPtr
          (ConstPtr logidPtr) -- NOTE: This is unsafe.
      handleOrtStatus ortApi ortStatusPtr $ do
        pure ()

foreign import capi unsafe
  "Onnxruntime/CApi_hsc.h _wrap_OrtApi_SetSessionLogId"
  _wrap_OrtApi_SetSessionLogId ::
    Ptr OrtSessionOptions ->
    ConstPtr CChar ->
    IO (Ptr OrtStatus)

#{def
  OrtStatus* _wrap_OrtApi_SetSessionLogId(
    HsOrtSessionOptions* options,
    const char* logid
  ) {
    return options->ortApi->SetSessionLogId(
      options->ortSessionOptions,
      logid
    );
  }
}

-------------------------------------------------------------------------------
-- OrtApi::SetSessionLogVerbosityLevel

{- |
> ORT_API2_STATUS(SetSessionLogVerbosityLevel,
>   _Inout_ OrtSessionOptions* options,
>   int session_log_verbosity_level
> );
-}
ortApiSetSessionLogVerbosityLevel ::
  OrtSessionOptions ->
  Int ->
  IO ()
ortApiSetSessionLogVerbosityLevel options sessionLogVerbosityLevel = do
  ortApi <- getOrtApi options
  withOrtSessionOptionsPtr options $ \optionsPtr -> do
    ortStatusPtr <-
      _wrap_OrtApi_SetSessionLogVerbosityLevel
        optionsPtr
        (fromIntegral sessionLogVerbosityLevel)
    handleOrtStatus ortApi ortStatusPtr $ do
      pure ()

foreign import capi unsafe
  "Onnxruntime/CApi_hsc.h _wrap_OrtApi_SetSessionLogVerbosityLevel"
  _wrap_OrtApi_SetSessionLogVerbosityLevel ::
    Ptr OrtSessionOptions ->
    CInt ->
    IO (Ptr OrtStatus)

#{def
  OrtStatus* _wrap_OrtApi_SetSessionLogVerbosityLevel(
    HsOrtSessionOptions* options,
    int sessionLogVerbosityLevel
  ) {
    return options->ortApi->SetSessionLogVerbosityLevel(
      options->ortSessionOptions,
      sessionLogVerbosityLevel
    );
  }
}

-------------------------------------------------------------------------------
-- OrtApi::SetSessionLogSeverityLevel

{-
> ORT_API2_STATUS(SetSessionLogSeverityLevel,
>   _Inout_ OrtSessionOptions* options,
>   int session_log_severity_level
> );
-}
ortApiSetSessionLogSeverityLevel ::
  OrtSessionOptions ->
  OrtLoggingLevel ->
  IO ()
ortApiSetSessionLogSeverityLevel options sessionLogSeverityLevel = do
  ortApi <- getOrtApi options
  withOrtSessionOptionsPtr options $ \optionsPtr -> do
    ortStatusPtr <-
      _wrap_OrtApi_SetSessionLogSeverityLevel
        optionsPtr
        sessionLogSeverityLevel
    handleOrtStatus ortApi ortStatusPtr $ do
      pure ()

foreign import capi unsafe
  "Onnxruntime/CApi_hsc.h _wrap_OrtApi_SetSessionLogSeverityLevel"
  _wrap_OrtApi_SetSessionLogSeverityLevel ::
    Ptr OrtSessionOptions ->
    OrtLoggingLevel ->
    IO (Ptr OrtStatus)

#{def
  OrtStatus* _wrap_OrtApi_SetSessionLogSeverityLevel(
    HsOrtSessionOptions* options,
    int sessionLogSeverityLevel
  ) {
    return options->ortApi->SetSessionLogSeverityLevel(
      options->ortSessionOptions,
      sessionLogSeverityLevel
    );
  }
}

-------------------------------------------------------------------------------
-- OrtApi::SetSessionGraphOptimizationLevel

{-
> ORT_API2_STATUS(SetSessionGraphOptimizationLevel,
>   _Inout_ OrtSessionOptions* options,
>   GraphOptimizationLevel graph_optimization_level
> );
-}
ortApiSetSessionGraphOptimizationLevel ::
  OrtSessionOptions ->
  GraphOptimizationLevel ->
  IO ()
ortApiSetSessionGraphOptimizationLevel options graphOptimizationlevel = do
  ortApi <- getOrtApi options
  withOrtSessionOptionsPtr options $ \optionsPtr -> do
    ortStatusPtr <-
      _wrap_OrtApi_SetSessionGraphOptimizationLevel
        optionsPtr
        graphOptimizationlevel
    handleOrtStatus ortApi ortStatusPtr $ do
      pure ()

foreign import capi unsafe
  "Onnxruntime/CApi_hsc.h _wrap_OrtApi_SetSessionGraphOptimizationLevel"
  _wrap_OrtApi_SetSessionGraphOptimizationLevel ::
    Ptr OrtSessionOptions ->
    GraphOptimizationLevel ->
    IO (Ptr OrtStatus)

#{def
  OrtStatus* _wrap_OrtApi_SetSessionGraphOptimizationLevel(
    HsOrtSessionOptions* options,
    int graphOptimizationlevel
  ) {
    return options->ortApi->SetSessionGraphOptimizationLevel(
      options->ortSessionOptions,
      graphOptimizationlevel
    );
  }
}

-------------------------------------------------------------------------------
-- OrtApi::SetIntraOpNumThreads

{- |
> ORT_API2_STATUS(SetIntraOpNumThreads,
>   _Inout_ OrtSessionOptions* options,
>   int intra_op_num_threads
> );
-}
ortApiSetIntraOpNumThreads ::
  OrtSessionOptions ->
  Int ->
  IO ()
ortApiSetIntraOpNumThreads options intraOpNumThreads = do
  ortApi <- getOrtApi options
  withOrtSessionOptionsPtr options $ \optionsPtr -> do
    ortStatusPtr <-
      _wrap_OrtApi_SetIntraOpNumThreads
        optionsPtr
        (fromIntegral intraOpNumThreads)
    handleOrtStatus ortApi ortStatusPtr $ do
      pure ()

foreign import capi unsafe
  "Onnxruntime/CApi_hsc.h _wrap_OrtApi_SetIntraOpNumThreads"
  _wrap_OrtApi_SetIntraOpNumThreads ::
    Ptr OrtSessionOptions ->
    CInt ->
    IO (Ptr OrtStatus)

#{def
  OrtStatus* _wrap_OrtApi_SetIntraOpNumThreads(
    HsOrtSessionOptions* options,
    int intraOpNumThreads
  ) {
    return options->ortApi->SetIntraOpNumThreads(
      options->ortSessionOptions,
      intraOpNumThreads
    );
  }
}

-------------------------------------------------------------------------------
-- OrtApi::SetInterOpNumThreads

{-
> ORT_API2_STATUS(SetInterOpNumThreads,
>   _Inout_ OrtSessionOptions* options,
>   int inter_op_num_threads
> );
-}
ortApiSetInterOpNumThreads ::
  OrtSessionOptions ->
  Int ->
  IO ()
ortApiSetInterOpNumThreads options interOpNumThreads = do
  ortApi <- getOrtApi options
  withOrtSessionOptionsPtr options $ \optionsPtr -> do
    ortStatusPtr <-
      _wrap_OrtApi_SetInterOpNumThreads
        optionsPtr
        (fromIntegral interOpNumThreads)
    handleOrtStatus ortApi ortStatusPtr $ do
      pure ()

foreign import capi unsafe
  "Onnxruntime/CApi_hsc.h _wrap_OrtApi_SetInterOpNumThreads"
  _wrap_OrtApi_SetInterOpNumThreads ::
    Ptr OrtSessionOptions ->
    CInt ->
    IO (Ptr OrtStatus)

#{def
  OrtStatus* _wrap_OrtApi_SetInterOpNumThreads(
    HsOrtSessionOptions* options,
    int interOpNumThreads
  ) {
    return options->ortApi->SetInterOpNumThreads(
      options->ortSessionOptions,
      interOpNumThreads
    );
  }
}

-------------------------------------------------------------------------------
-- OrtApi::CreateCustomOpDomain

-- ORT_API2_STATUS(CreateCustomOpDomain, _In_ const char* domain, _Outptr_ OrtCustomOpDomain** out);

-- TODO: unimplemented

-------------------------------------------------------------------------------
-- OrtApi::CustomOpDomain_Add

-- ORT_API2_STATUS(CustomOpDomain_Add, _Inout_ OrtCustomOpDomain* custom_op_domain, _In_ const OrtCustomOp* op);

-- TODO: unimplemented

-------------------------------------------------------------------------------
-- OrtApi::AddCustomOpDomain

-- ORT_API2_STATUS(AddCustomOpDomain, _Inout_ OrtSessionOptions* options, _In_ OrtCustomOpDomain* custom_op_domain);

-- TODO: unimplemented

-------------------------------------------------------------------------------
-- OrtApi::RegisterCustomOpsLibrary

-- ORT_API2_STATUS(RegisterCustomOpsLibrary, _Inout_ OrtSessionOptions* options, _In_ const char* library_path, _Outptr_ void** library_handle);

-- TODO: unimplemented

-------------------------------------------------------------------------------
-- OrtApi::SessionGetInputCount

{- |
> ORT_API2_STATUS(SessionGetInputCount,
> _In_ const OrtSession* session,
> _Out_ size_t* out
> );
-}
ortApiSessionGetInputCount ::
  OrtSession ->
  IO Word64
ortApiSessionGetInputCount ortSession = do
  ortApi <- getOrtApi ortSession
  withOrtSessionPtr ortSession $ \ortSessionPtr ->
    alloca $ \outPtr -> do
      ortStatusPtr <-
        _wrap_OrtApi_SessionGetInputCount
          (ConstPtr ortSessionPtr)
          outPtr
      handleOrtStatus ortApi ortStatusPtr $ do
        CSize inputCount <- peek outPtr
        pure inputCount

foreign import capi unsafe
  "Onnxruntime/CApi_hsc.h _wrap_OrtApi_SessionGetInputCount"
  _wrap_OrtApi_SessionGetInputCount ::
    ConstPtr OrtSession ->
    Ptr CSize ->
    IO (Ptr OrtStatus)

#{def
  OrtStatus* _wrap_OrtApi_SessionGetInputCount(
    const HsOrtSession* ortSession,
    size_t* out
  ) {
    return ortSession->ortApi->SessionGetInputCount(
      ortSession->ortSession,
      out
    );
  }
}

-------------------------------------------------------------------------------
-- OrtApi::SessionGetOutputCount

{- |
> ORT_API2_STATUS(SessionGetOutputCount,
> _In_ const OrtSession* session,
> _Out_ size_t* out
> );
-}
ortApiSessionGetOutputCount ::
  OrtSession ->
  IO Word64
ortApiSessionGetOutputCount ortSession = do
  ortApi <- getOrtApi ortSession
  withOrtSessionPtr ortSession $ \ortSessionPtr ->
    alloca $ \outPtr -> do
      ortStatusPtr <-
        _wrap_OrtApi_SessionGetOutputCount
          (ConstPtr ortSessionPtr)
          outPtr
      handleOrtStatus ortApi ortStatusPtr $ do
        CSize inputCount <- peek outPtr
        pure inputCount

foreign import capi unsafe
  "Onnxruntime/CApi_hsc.h _wrap_OrtApi_SessionGetOutputCount"
  _wrap_OrtApi_SessionGetOutputCount ::
    ConstPtr OrtSession ->
    Ptr CSize ->
    IO (Ptr OrtStatus)

#{def
  OrtStatus* _wrap_OrtApi_SessionGetOutputCount(
    const HsOrtSession* ortSession,
    size_t* out
  ) {
    return ortSession->ortApi->SessionGetOutputCount(
      ortSession->ortSession,
      out
    );
  }
}

-------------------------------------------------------------------------------
-- OrtApi::SessionGetOverridableInitializerCount

-- ORT_API2_STATUS(SessionGetOverridableInitializerCount, _In_ const OrtSession* session, _Out_ size_t* out);

-- TODO: unimplemented

-------------------------------------------------------------------------------
-- OrtApi::SessionGetInputTypeInfo

{- |
> ORT_API2_STATUS(SessionGetInputTypeInfo,
> _In_ const OrtSession* session,
> size_t index,
> _Outptr_ OrtTypeInfo** type_info
> );
-}
ortApiSessionGetInputTypeInfo ::
  OrtSession ->
  Word64 ->
  IO OrtTypeInfo
ortApiSessionGetInputTypeInfo ortSession index = do
  ortApi <- getOrtApi ortSession
  withOrtSessionPtr ortSession $ \ortSessionPtr ->
    alloca $ \outPtr -> do
      ortStatusPtr <-
        _wrap_OrtApi_SessionGetInputTypeInfo
          (ConstPtr ortSessionPtr)
          (CSize index)
          outPtr
      handleOrtStatus ortApi ortStatusPtr $
        wrapCOrtTypeInfo ortApi
          =<< peek outPtr

foreign import capi unsafe
  "Onnxruntime/CApi_hsc.h _wrap_OrtApi_SessionGetInputTypeInfo"
  _wrap_OrtApi_SessionGetInputTypeInfo ::
    ConstPtr OrtSession ->
    CSize ->
    Ptr (Ptr COrtTypeInfo) ->
    IO (Ptr OrtStatus)

#{def
  OrtStatus* _wrap_OrtApi_SessionGetInputTypeInfo(
    const HsOrtSession* ortSession,
    size_t index,
    COrtTypeInfo** out
  ) {
    return ortSession->ortApi->SessionGetInputTypeInfo(
      ortSession->ortSession,
      index,
      out
    );
  }
}

-------------------------------------------------------------------------------
-- OrtApi::SessionGetOutputTypeInfo

{- |
> ORT_API2_STATUS(SessionGetOutputTypeInfo,
> _In_ const OrtSession* session,
> size_t index,
> _Outptr_ OrtTypeInfo** type_info
> );
-}
ortApiSessionGetOutputTypeInfo ::
  OrtSession ->
  Word64 ->
  IO OrtTypeInfo
ortApiSessionGetOutputTypeInfo ortSession index = do
  ortApi <- getOrtApi ortSession
  withOrtSessionPtr ortSession $ \ortSessionPtr ->
    alloca $ \outPtr -> do
      ortStatusPtr <-
        _wrap_OrtApi_SessionGetOutputTypeInfo
          (ConstPtr ortSessionPtr)
          (CSize index)
          outPtr
      handleOrtStatus ortApi ortStatusPtr $
        wrapCOrtTypeInfo ortApi
          =<< peek outPtr

foreign import capi unsafe
  "Onnxruntime/CApi_hsc.h _wrap_OrtApi_SessionGetOutputTypeInfo"
  _wrap_OrtApi_SessionGetOutputTypeInfo ::
    ConstPtr OrtSession ->
    CSize ->
    Ptr (Ptr COrtTypeInfo) ->
    IO (Ptr OrtStatus)

#{def
  OrtStatus* _wrap_OrtApi_SessionGetOutputTypeInfo(
    const HsOrtSession* ortSession,
    size_t index,
    COrtTypeInfo** out
  ) {
    return ortSession->ortApi->SessionGetOutputTypeInfo(
      ortSession->ortSession,
      index,
      out
    );
  }
}

-------------------------------------------------------------------------------
-- OrtApi::SessionGetOverridableInitializerTypeInfo

-- ORT_API2_STATUS(SessionGetOverridableInitializerTypeInfo, _In_ const OrtSession* session, size_t index, _Outptr_ OrtTypeInfo** type_info);

-- TODO: unimplemented

-------------------------------------------------------------------------------
-- OrtApi::SessionGetInputName

{- |
> ORT_API2_STATUS(SessionGetInputName,
>   _In_ const OrtSession* session,
>   size_t index,
>   _Inout_ OrtAllocator* allocator,
>   _Outptr_ char** value
> );
-}
ortApiSessionGetInputName ::
  OrtSession ->
  Word64 ->
  OrtAllocator ->
  IO String
ortApiSessionGetInputName ortSession index allocator = do
  ortApi <- getOrtApi ortSession
  withOrtSessionPtr ortSession $ \ortSessionPtr ->
    withOrtAllocatorPtr allocator $ \allocatorPtr -> do
      alloca $ \valuePtr -> do
        ortStatusPtr <-
          _wrap_OrtApi_SessionGetInputName
            (ConstPtr ortSessionPtr)
            (CSize index)
            allocatorPtr
            valuePtr
        handleOrtStatus ortApi ortStatusPtr $ do
          cString <- peek valuePtr
          result <- peekCString cString
          ortApiAllocatorFree allocator cString
          pure result

foreign import capi unsafe
  "Onnxruntime/CApi_hsc.h _wrap_OrtApi_SessionGetInputName"
  _wrap_OrtApi_SessionGetInputName ::
    ConstPtr OrtSession ->
    CSize ->
    Ptr OrtAllocator ->
    Ptr CString ->
    IO (Ptr OrtStatus)

#{def
  OrtStatus* _wrap_OrtApi_SessionGetInputName(
    const HsOrtSession* ortSession,
    size_t index,
    HsOrtAllocator* allocator,
    char** value
  ) {
    return ortSession->ortApi->SessionGetInputName(
      ortSession->ortSession,
      index,
      allocator->ortAllocator,
      value
    );
  }
}

-------------------------------------------------------------------------------
-- OrtApi::SessionGetOutputName

{- |
> ORT_API2_STATUS(SessionGetOutputName,
>   _In_ const OrtSession* session,
>   size_t index,
>   _Inout_ OrtAllocator* allocator,
>   _Outptr_ char** value
> );
-}
ortApiSessionGetOutputName ::
  OrtSession ->
  Word64 ->
  OrtAllocator ->
  IO String
ortApiSessionGetOutputName ortSession index allocator = do
  ortApi <- getOrtApi ortSession
  withOrtSessionPtr ortSession $ \ortSessionPtr ->
    withOrtAllocatorPtr allocator $ \allocatorPtr -> do
      alloca $ \valuePtr -> do
        ortStatusPtr <-
          _wrap_OrtApi_SessionGetOutputName
            (ConstPtr ortSessionPtr)
            (CSize index)
            allocatorPtr
            valuePtr
        handleOrtStatus ortApi ortStatusPtr $ do
          cString <- peek valuePtr
          result <- peekCString cString
          ortApiAllocatorFree allocator cString
          pure result

foreign import capi unsafe
  "Onnxruntime/CApi_hsc.h _wrap_OrtApi_SessionGetOutputName"
  _wrap_OrtApi_SessionGetOutputName ::
    ConstPtr OrtSession ->
    CSize ->
    Ptr OrtAllocator ->
    Ptr CString ->
    IO (Ptr OrtStatus)

#{def
  OrtStatus* _wrap_OrtApi_SessionGetOutputName(
    const HsOrtSession* ortSession,
    size_t index,
    HsOrtAllocator* allocator,
    char** value
  ) {
    return ortSession->ortApi->SessionGetOutputName(
      ortSession->ortSession,
      index,
      allocator->ortAllocator,
      value
    );
  }
}


-------------------------------------------------------------------------------
-- OrtApi::SessionGetOverridableInitializerName

-- ORT_API2_STATUS(SessionGetOverridableInitializerName, _In_ const OrtSession* session, size_t index, _Inout_ OrtAllocator* allocator, _Outptr_ char** value);

-- TODO: unimplemented

-------------------------------------------------------------------------------
-- OrtApi::CreateRunOptions

{- |
> ORT_API2_STATUS(CreateRunOptions,
>   _Outptr_ OrtRunOptions** options
> );
-}
ortApiCreateRunOptions ::
  OrtApi ->
  IO OrtRunOptions
ortApiCreateRunOptions ortApi = do
  alloca $ \outPtr -> do
    ortStatusPtr <-
      _wrap_OrtApi_CreateRunOptions
        ortApi.ortApiConstPtr
        outPtr
    handleOrtStatus ortApi ortStatusPtr $ do
      cOrtRunOptionsPtr <- peek outPtr
      wrapCOrtRunOptions ortApi cOrtRunOptionsPtr

foreign import capi unsafe
  "Onnxruntime/CApi_hsc.h _wrap_OrtApi_CreateRunOptions"
  _wrap_OrtApi_CreateRunOptions ::
    ConstPtr OrtApi ->
    Ptr (Ptr COrtRunOptions) ->
    IO (Ptr OrtStatus)

#{def
  OrtStatus* _wrap_OrtApi_CreateRunOptions(
    const OrtApi* ortApi,
    COrtRunOptions** out
  ) {
    return ortApi->CreateRunOptions(out);
  }
}

-------------------------------------------------------------------------------
-- OrtApi::RunOptionsSetRunLogVerbosityLevel

{- |
> ORT_API2_STATUS(RunOptionsSetRunLogVerbosityLevel,
>   _Inout_ OrtRunOptions* options,
>   int log_verbosity_level
> );
-}
ortApiRunOptionsSetRunLogVerbosityLevel ::
  OrtRunOptions ->
  Int ->
  IO ()
ortApiRunOptionsSetRunLogVerbosityLevel options logVerbosityLevel = do
  ortApi <- getOrtApi options
  withOrtRunOptionsPtr options $ \optionsPtr -> do
    ortStatusPtr <-
      _wrap_OrtApi_RunOptionsSetRunLogVerbosityLevel
        optionsPtr
        (fromIntegral logVerbosityLevel)
    handleOrtStatus ortApi ortStatusPtr $ do
      pure ()

foreign import capi unsafe
  "Onnxruntime/CApi_hsc.h _wrap_OrtApi_RunOptionsSetRunLogVerbosityLevel"
  _wrap_OrtApi_RunOptionsSetRunLogVerbosityLevel ::
    Ptr OrtRunOptions ->
    CInt ->
    IO (Ptr OrtStatus)

#{def
  OrtStatus* _wrap_OrtApi_RunOptionsSetRunLogVerbosityLevel(
    HsOrtRunOptions* options,
    int logVerbosityLevel
  ) {
    return options->ortApi->RunOptionsSetRunLogVerbosityLevel(
      options->ortRunOptions,
      logVerbosityLevel
    );
  }
}

-------------------------------------------------------------------------------
-- OrtApi::RunOptionsSetRunLogSeverityLevel

{- |
> ORT_API2_STATUS(RunOptionsSetRunLogSeverityLevel,
>   _Inout_ OrtRunOptions* options,
>   int log_severity_level
> );
-}
ortApiRunOptionsSetRunLogSeverityLevel ::
  OrtRunOptions ->
  Int ->
  IO ()
ortApiRunOptionsSetRunLogSeverityLevel options logSeverityLevel = do
  ortApi <- getOrtApi options
  withOrtRunOptionsPtr options $ \optionsPtr -> do
    ortStatusPtr <-
      _wrap_OrtApi_RunOptionsSetRunLogSeverityLevel
        optionsPtr
        (fromIntegral logSeverityLevel)
    handleOrtStatus ortApi ortStatusPtr $ do
      pure ()

foreign import capi unsafe
  "Onnxruntime/CApi_hsc.h _wrap_OrtApi_RunOptionsSetRunLogSeverityLevel"
  _wrap_OrtApi_RunOptionsSetRunLogSeverityLevel ::
    Ptr OrtRunOptions ->
    CInt ->
    IO (Ptr OrtStatus)

#{def
  OrtStatus* _wrap_OrtApi_RunOptionsSetRunLogSeverityLevel(
    HsOrtRunOptions* options,
    int logSeverityLevel
  ) {
    return options->ortApi->RunOptionsSetRunLogSeverityLevel(
      options->ortRunOptions,
      logSeverityLevel
    );
  }
}

-------------------------------------------------------------------------------
-- OrtApi::RunOptionsSetRunTag

{- |
> ORT_API2_STATUS(RunOptionsSetRunTag,
>   _Inout_ OrtRunOptions* options,
>   _In_ const char* run_tag
> );
-}
ortApiRunOptionsSetRunTag ::
  OrtRunOptions ->
  String ->
  IO ()
ortApiRunOptionsSetRunTag options runTag = do
  ortApi <- getOrtApi options
  withCString runTag $ \runTagPtr -> do
    withOrtRunOptionsPtr options $ \optionsPtr -> do
      ortStatusPtr <-
        _wrap_OrtApi_RunOptionsSetRunTag
          optionsPtr
          (ConstPtr runTagPtr) -- NOTE: This is unsafe.
      handleOrtStatus ortApi ortStatusPtr $ do
        pure ()

foreign import capi unsafe
  "Onnxruntime/CApi_hsc.h _wrap_OrtApi_RunOptionsSetRunTag"
  _wrap_OrtApi_RunOptionsSetRunTag ::
    Ptr OrtRunOptions ->
    ConstPtr CChar ->
    IO (Ptr OrtStatus)

#{def
  OrtStatus* _wrap_OrtApi_RunOptionsSetRunTag(
    HsOrtRunOptions* options,
    const char* runTag
  ) {
    return options->ortApi->RunOptionsSetRunTag(
      options->ortRunOptions,
      runTag
    );
  }
}

-------------------------------------------------------------------------------
-- OrtApi::RunOptionsGetRunLogVerbosityLevel

{- |
> ORT_API2_STATUS(RunOptionsGetRunLogVerbosityLevel,
>   _In_ const OrtRunOptions* options,
>   _Out_ int* log_verbosity_level
> );
-}
ortApiRunOptionsGetRunLogVerbosityLevel ::
  OrtRunOptions ->
  IO Int
ortApiRunOptionsGetRunLogVerbosityLevel options = do
  ortApi <- getOrtApi options
  alloca $ \outPtr -> do
    withOrtRunOptionsPtr options $ \optionsPtr -> do
      ortStatusPtr <-
        _wrap_OrtApi_RunOptionsGetRunLogVerbosityLevel
          optionsPtr
          outPtr
      handleOrtStatus ortApi ortStatusPtr $ do
        fromIntegral <$> peek outPtr

foreign import capi unsafe
  "Onnxruntime/CApi_hsc.h _wrap_OrtApi_RunOptionsGetRunLogVerbosityLevel"
  _wrap_OrtApi_RunOptionsGetRunLogVerbosityLevel ::
    Ptr OrtRunOptions ->
    Ptr CInt ->
    IO (Ptr OrtStatus)

#{def
  OrtStatus* _wrap_OrtApi_RunOptionsGetRunLogVerbosityLevel(
    HsOrtRunOptions* options,
    int* logVerbosityLevel
  ) {
    return options->ortApi->RunOptionsGetRunLogVerbosityLevel(
      options->ortRunOptions,
      logVerbosityLevel
    );
  }
}

-------------------------------------------------------------------------------
-- OrtApi::RunOptionsGetRunLogSeverityLevel

{- |
> ORT_API2_STATUS(RunOptionsGetRunLogSeverityLevel,
>   _In_ const OrtRunOptions* options,
>   _Out_ int* log_severity_level
> );
-}
ortApiRunOptionsGetRunLogSeverityLevel ::
  OrtRunOptions ->
  IO Int
ortApiRunOptionsGetRunLogSeverityLevel options = do
  ortApi <- getOrtApi options
  alloca $ \outPtr -> do
    withOrtRunOptionsPtr options $ \optionsPtr -> do
      ortStatusPtr <-
        _wrap_OrtApi_RunOptionsGetRunLogSeverityLevel
          optionsPtr
          outPtr
      handleOrtStatus ortApi ortStatusPtr $ do
        fromIntegral <$> peek outPtr

foreign import capi unsafe
  "Onnxruntime/CApi_hsc.h _wrap_OrtApi_RunOptionsGetRunLogSeverityLevel"
  _wrap_OrtApi_RunOptionsGetRunLogSeverityLevel ::
    Ptr OrtRunOptions ->
    Ptr CInt ->
    IO (Ptr OrtStatus)

#{def
  OrtStatus* _wrap_OrtApi_RunOptionsGetRunLogSeverityLevel(
    HsOrtRunOptions* options,
    int* logSeverityLevel
  ) {
    return options->ortApi->RunOptionsGetRunLogSeverityLevel(
      options->ortRunOptions,
      logSeverityLevel
    );
  }
}

-------------------------------------------------------------------------------
-- OrtApi::RunOptionsGetRunTag

{- |
> ORT_API2_STATUS(RunOptionsGetRunTag,
>   _In_ const OrtRunOptions* options,
>   _Out_ const char** run_tag
> );
-}
ortApiRunOptionsGetRunTag ::
  OrtRunOptions ->
  IO String
ortApiRunOptionsGetRunTag options = do
  ortApi <- getOrtApi options
  alloca @(ConstPtr CChar) $ \outPtr -> do
    withOrtRunOptionsPtr options $ \optionsPtr -> do
      ortStatusPtr <-
        _wrap_OrtApi_RunOptionsGetRunTag
          optionsPtr
          outPtr
      handleOrtStatus ortApi ortStatusPtr $ do
        peekCString . unConstPtr
          =<< peek outPtr

foreign import capi unsafe
  "Onnxruntime/CApi_hsc.h _wrap_OrtApi_RunOptionsGetRunTag"
  _wrap_OrtApi_RunOptionsGetRunTag ::
    Ptr OrtRunOptions ->
    Ptr (ConstPtr CChar)  ->
    IO (Ptr OrtStatus)

#{def
  OrtStatus* _wrap_OrtApi_RunOptionsGetRunTag(
    HsOrtRunOptions* options,
    const char** runTag
  ) {
    return options->ortApi->RunOptionsGetRunTag(
      options->ortRunOptions,
      runTag
    );
  }
}

-------------------------------------------------------------------------------
-- OrtApi::RunOptionsSetTerminate

{- |
> ORT_API2_STATUS(RunOptionsSetTerminate,
>   _Inout_ OrtRunOptions* options
> );
-}
ortApiRunOptionsSetTerminate ::
  OrtRunOptions ->
  IO ()
ortApiRunOptionsSetTerminate options = do
  ortApi <- getOrtApi options
  withOrtRunOptionsPtr options $ \optionsPtr -> do
    ortStatusPtr <-
      _wrap_OrtApi_RunOptionsSetTerminate
        optionsPtr
    handleOrtStatus ortApi ortStatusPtr $ do
      pure ()

foreign import capi unsafe
  "Onnxruntime/CApi_hsc.h _wrap_OrtApi_RunOptionsSetTerminate"
  _wrap_OrtApi_RunOptionsSetTerminate ::
    Ptr OrtRunOptions ->
    IO (Ptr OrtStatus)

#{def
  OrtStatus* _wrap_OrtApi_RunOptionsSetTerminate(
    HsOrtRunOptions* options
  ) {
    return options->ortApi->RunOptionsSetTerminate(
      options->ortRunOptions
    );
  }
}

-------------------------------------------------------------------------------
-- OrtApi::RunOptionsUnsetTerminate

{- |
> ORT_API2_STATUS(RunOptionsUnsetTerminate,
>   _Inout_ OrtRunOptions* options
> );
-}
ortApiRunOptionsUnsetTerminate ::
  OrtRunOptions ->
  IO ()
ortApiRunOptionsUnsetTerminate options = do
  ortApi <- getOrtApi options
  withOrtRunOptionsPtr options $ \optionsPtr -> do
    ortStatusPtr <-
      _wrap_OrtApi_RunOptionsUnsetTerminate
        optionsPtr
    handleOrtStatus ortApi ortStatusPtr $ do
      pure ()

foreign import capi unsafe
  "Onnxruntime/CApi_hsc.h _wrap_OrtApi_RunOptionsUnsetTerminate"
  _wrap_OrtApi_RunOptionsUnsetTerminate ::
    Ptr OrtRunOptions ->
    IO (Ptr OrtStatus)

#{def
  OrtStatus* _wrap_OrtApi_RunOptionsUnsetTerminate(
    HsOrtRunOptions* options
  ) {
    return options->ortApi->RunOptionsUnsetTerminate(
      options->ortRunOptions
    );
  }
}

-------------------------------------------------------------------------------
-- OrtApi::CreateTensorAsOrtValue

-- TODO: Required to get OrtApi::Run working.

{-
> ORT_API2_STATUS(CreateTensorAsOrtValue,
>   _Inout_ OrtAllocator* allocator,
>   _In_ const int64_t* shape,
>   size_t shape_len,
>   ONNXTensorElementDataType type,
>   _Outptr_ OrtValue** out
> );
-}
ortApiCreateTensorAsOrtValue ::
  OrtAllocator ->
  [Int64] ->
  ONNXTensorElementDataType ->
  IO OrtValue
ortApiCreateTensorAsOrtValue allocator shape dataType = do
  ortApi <- getOrtApi allocator
  withCTypePtr allocator $ \cOrtAllocatorPtr -> do
    withArrayLen shape $ \shapeLen shapePtr -> do
      alloca $ \outPtr -> do
        ortStatusPtr <-
          _wrap_OrtApi_CreateTensorAsOrtValue
            ortApi.ortApiConstPtr
            cOrtAllocatorPtr
            shapePtr
            (fromIntegral shapeLen)
            dataType
            outPtr
        handleOrtStatus ortApi ortStatusPtr $ do
          wrapCOrtValue ortApi
            =<< peek outPtr

foreign import capi unsafe
  "Onnxruntime/CApi_hsc.h _wrap_OrtApi_CreateTensorAsOrtValue"
  _wrap_OrtApi_CreateTensorAsOrtValue ::
    ConstPtr OrtApi ->
    Ptr COrtAllocator ->
    Ptr ( #{type int64_t} ) ->
    ( #{type size_t} ) ->
    ONNXTensorElementDataType ->
    Ptr (Ptr COrtValue) ->
    IO (Ptr OrtStatus)

#{def
  OrtStatus* _wrap_OrtApi_CreateTensorAsOrtValue(
    const OrtApi* ortApi,
    COrtAllocator* allocator,
    const int64_t* shape,
    size_t shape_len,
    ONNXTensorElementDataType type,
    COrtValue** out
  ) {
    return ortApi->CreateTensorAsOrtValue(
      allocator,
      shape,
      shape_len,
      type,
      out
    );
  }
}

-------------------------------------------------------------------------------
-- OrtApi::CreateTensorWithDataAsOrtValue

{-
> ORT_API2_STATUS(CreateTensorWithDataAsOrtValue,
>   _In_ const OrtMemoryInfo* info,
>   _Inout_ void* p_data,
>   size_t p_data_len,
>   _In_ const int64_t* shape,
>   size_t shape_len,
>   ONNXTensorElementDataType type,
>   _Outptr_ OrtValue** out
> );
-}
ortApiWithTensorWithDataAsOrtValue ::
  forall a b.
  (IsONNXTensorElementDataType a) =>
  OrtApi ->
  Vector a ->
  [Int64] ->
  (OrtValue -> IO b) ->
  IO b
ortApiWithTensorWithDataAsOrtValue ortApi values shape action = do
  memoryInfo <- ortApiCreateCpuMemoryInfo ortApi OrtDeviceAllocator OrtMemTypeCPU
  withCTypePtr memoryInfo $ \cOrtMemoryInfoPtr -> do
    let valueLen = VS.length values
    VS.unsafeWith values $ \valuePtr -> do
      withArrayLen shape $ \shapeLen shapePtr ->
        alloca $ \outPtr -> do
          ortStatusPtr <-
            _wrap_OrtApi_CreateTensorWithDataAsOrtValue
              ortApi
              cOrtMemoryInfoPtr
              (castPtr valuePtr)
              (fromIntegral $ valueLen * sizeOf (undefined :: a))
              shapePtr
              (fromIntegral shapeLen)
              (getONNXTensorElementDataType (Proxy :: Proxy a))
              outPtr
          ortValue <-
            handleOrtStatus ortApi ortStatusPtr $ do
              wrapCOrtValue ortApi
                =<< peek outPtr
          action ortValue

foreign import capi unsafe
  "Onnxruntime/CApi_hsc.h _wrap_OrtApi_CreateTensorWithDataAsOrtValue"
  _wrap_OrtApi_CreateTensorWithDataAsOrtValue ::
    OrtApi ->
    Ptr COrtMemoryInfo ->
    Ptr Void ->
    ( #{type size_t} ) ->
    Ptr ( #{type int64_t} ) ->
    ( #{type size_t} ) ->
    ONNXTensorElementDataType ->
    Ptr (Ptr COrtValue) ->
    IO (Ptr OrtStatus)

#{def
  OrtStatus* _wrap_OrtApi_CreateTensorWithDataAsOrtValue(
    const OrtApi* ortApi,
    const COrtMemoryInfo* info,
    void* p_data,
    size_t p_data_len,
    const int64_t* shape,
    size_t shapeLen,
    ONNXTensorElementDataType type,
    COrtValue** out
  ) {
    return ortApi->CreateTensorWithDataAsOrtValue(
      info,
      p_data,
      p_data_len,
      shape,
      shapeLen,
      type,
      out
    );
  }
}

-------------------------------------------------------------------------------
-- OrtApi::IsTensor

{-
> ORT_API2_STATUS(IsTensor,
>   _In_ const OrtValue* value,
>   _Out_ int* out
> );
-}
ortApiIsTensor ::
  OrtValue ->
  IO Bool
ortApiIsTensor ortValue = do
  ortApi <- getOrtApi ortValue
  withOrtValuePtr ortValue $ \ortValuePtr ->
    alloca $ \outPtr -> do
      ortStatusPtr <-
        _wrap_OrtApi_IsTensor
          ortValuePtr
          outPtr
      handleOrtStatus ortApi ortStatusPtr $ do
        (==1) <$> peek outPtr

foreign import capi unsafe
  "Onnxruntime/CApi_hsc.h _wrap_OrtApi_IsTensor"
  _wrap_OrtApi_IsTensor ::
    Ptr OrtValue ->
    Ptr ( #{type int} ) ->
    IO (Ptr OrtStatus)

#{def
  OrtStatus* _wrap_OrtApi_IsTensor(
    const HsOrtValue* value,
    int* out
  ) {
    return value->ortApi->IsTensor(
      value->ortValue,
      out
    );
  }
}

-------------------------------------------------------------------------------
-- OrtApi::GetTensorMutableData

data ONNXTypeError
  = ErrONNXTypeMismatch
    -- | Expected type.
    !ONNXType
    -- | Actual type.
    !ONNXType
  | ErrONNXTensorElementDataTypeMismatch
    -- | Expected element data type.
    !ONNXTensorElementDataType
    -- | Actual element data type.
    !ONNXTensorElementDataType
  deriving (Eq, Show)

instance Exception ONNXTypeError

ortApiCheckType ::
  ONNXType ->
  OrtValue ->
  IO ()
ortApiCheckType expectedType ortValue = do
  actualType <- ortApiGetValueType ortValue
  unless (expectedType == actualType) $
    throwIO (ErrONNXTypeMismatch expectedType actualType)

ortApiCheckTensorElementDataType ::
  ONNXTensorElementDataType ->
  OrtValue ->
  IO ()
ortApiCheckTensorElementDataType expectedElementType ortValue = do
  ortApiCheckType ONNXTypeTensor ortValue
  tensorTypeAndShape <- ortApiGetTensorTypeAndShape ortValue
  actualElementType <- ortApiGetTensorElementType tensorTypeAndShape
  unless (expectedElementType == actualElementType) $
    throwIO (ErrONNXTensorElementDataTypeMismatch expectedElementType actualElementType)

{-
> ORT_API2_STATUS(GetTensorMutableData,
>   _In_ OrtValue* value,
>   _Outptr_ void** out
> );
-}
ortApiWithTensorData ::
  forall a b.
  (IsONNXTensorElementDataType a) =>
  OrtValue ->
  (Vector a -> IO b) ->
  IO b
ortApiWithTensorData ortValue action = do
  ortApi <- getOrtApi ortValue
  -- Check the tensor type
  ortApiCheckTensorElementDataType (getONNXTensorElementDataType (Proxy :: Proxy a)) ortValue
  -- Get the tensor dimensions
  tensorTypeAndShape <- ortApiGetTensorTypeAndShape ortValue
  tensorElementCount <- ortApiGetTensorShapeElementCount tensorTypeAndShape
  -- Get the tensor data
  withOrtValuePtr ortValue $ \ortValuePtr ->
    alloca $ \outPtr -> do
      ortStatusPtr <-
        _wrap_OrtApi_GetTensorMutableData
          ortValuePtr
          outPtr
      mutableDataPtr <-
        handleOrtStatus ortApi ortStatusPtr $ do
          castPtr
            <$> peek outPtr
      mutableDataForeignPtr <-
        newForeignPtr_ mutableDataPtr
      action $
        VS.unsafeFromForeignPtr0 mutableDataForeignPtr (fromIntegral tensorElementCount)

foreign import capi unsafe
  "Onnxruntime/CApi_hsc.h _wrap_OrtApi_GetTensorMutableData"
  _wrap_OrtApi_GetTensorMutableData ::
    Ptr OrtValue ->
    Ptr (Ptr Void) ->
    IO (Ptr OrtStatus)

#{def
  OrtStatus* _wrap_OrtApi_GetTensorMutableData(
    const HsOrtValue* value,
    void** out
  ) {
    return value->ortApi->GetTensorMutableData(
      value->ortValue,
      out
    );
  }
}

-------------------------------------------------------------------------------
-- OrtApi::FillStringTensor

{-
> ORT_API2_STATUS(FillStringTensor,
>   _Inout_ OrtValue* value,
>   _In_ const char* const* s,
>   size_t s_len
> );
-}

-- TODO: unimplemented

-------------------------------------------------------------------------------
-- OrtApi::GetStringTensorDataLength

{-
> ORT_API2_STATUS(GetStringTensorDataLength,
>   _In_ const OrtValue* value,
>   _Out_ size_t* len
> );
-}

-- TODO: unimplemented

-------------------------------------------------------------------------------
-- OrtApi::GetStringTensorContent

{-
> ORT_API2_STATUS(GetStringTensorContent,
>   _In_ const OrtValue* value,
>   _Out_writes_bytes_all_(s_len) void* s,
>   size_t s_len,
>   _Out_writes_all_(offsets_len) size_t* offsets,
>   size_t offsets_len
> );
-}

-- TODO: unimplemented

-------------------------------------------------------------------------------
-- OrtApi::CastTypeInfoToTensorInfo

{-
> ORT_API2_STATUS(CastTypeInfoToTensorInfo,
>   _In_ const OrtTypeInfo* type_info,
>   _Outptr_result_maybenull_ const OrtTensorTypeAndShapeInfo** out
> );
-}
ortApiCastTypeInfoToTensorInfo ::
  OrtTypeInfo ->
  IO OrtTensorTypeAndShapeInfo
ortApiCastTypeInfoToTensorInfo ortTypeInfo = do
  ortApi <- getOrtApi ortTypeInfo
  withOrtTypeInfoPtr ortTypeInfo $ \ortTypeInfoPtr ->
    alloca $ \outPtr -> do
      ortStatusPtr <-
        _wrap_OrtApi_CastTypeInfoToTensorInfo
          (ConstPtr ortTypeInfoPtr)
          outPtr
      handleOrtStatus ortApi ortStatusPtr $ do
        ConstPtr ortTypeAndShapeInfoPtr <- peek outPtr
        -- If the ortTypeInfo does not represent a tensor type,
        -- then OrtApi::CastTypeInfoToTensorInfo returns NULL.
        if ortTypeAndShapeInfoPtr == nullPtr
          then do
            actualType <- ortApiGetOnnxTypeFromTypeInfo ortTypeInfo
            throwIO (ErrONNXTypeMismatch ONNXTypeTensor actualType)
          else
            wrapCOrtTensorTypeAndShapeInfoFromOrtTypeInfo ortApi ortTypeInfo ortTypeAndShapeInfoPtr

foreign import capi unsafe
  "Onnxruntime/CApi_hsc.h _wrap_OrtApi_CastTypeInfoToTensorInfo"
  _wrap_OrtApi_CastTypeInfoToTensorInfo ::
    ConstPtr OrtTypeInfo ->
    Ptr (ConstPtr COrtTensorTypeAndShapeInfo) ->
    IO (Ptr OrtStatus)

#{def
  OrtStatus* _wrap_OrtApi_CastTypeInfoToTensorInfo(
    const HsOrtTypeInfo* value,
    const COrtTensorTypeAndShapeInfo** out
  ) {
    return value->ortApi->CastTypeInfoToTensorInfo(
      value->ortTypeInfo,
      out
    );
  }
}

-------------------------------------------------------------------------------
-- OrtApi::GetOnnxTypeFromTypeInfo

{-
> ORT_API2_STATUS(GetOnnxTypeFromTypeInfo,
>   _In_ const OrtTypeInfo* type_info,
>   _Out_ enum ONNXType* out
> );
-}
ortApiGetOnnxTypeFromTypeInfo ::
  OrtTypeInfo ->
  IO ONNXType
ortApiGetOnnxTypeFromTypeInfo ortTypeInfo = do
  ortApi <- getOrtApi ortTypeInfo
  withOrtTypeInfoPtr ortTypeInfo $ \ortTypeInfoPtr ->
    alloca $ \outPtr -> do
      ortStatusPtr <-
        _wrap_OrtApi_GetOnnxTypeFromTypeInfo
          (ConstPtr ortTypeInfoPtr)
          outPtr
      handleOrtStatus ortApi ortStatusPtr $
        ONNXType
          <$> peek outPtr

foreign import capi unsafe
  "Onnxruntime/CApi_hsc.h _wrap_OrtApi_GetOnnxTypeFromTypeInfo"
  _wrap_OrtApi_GetOnnxTypeFromTypeInfo ::
    ConstPtr OrtTypeInfo ->
    Ptr ( #{type ONNXType} ) ->
    IO (Ptr OrtStatus)

#{def
  OrtStatus* _wrap_OrtApi_GetOnnxTypeFromTypeInfo(
    const HsOrtTypeInfo* value,
    ONNXType* out
  ) {
    return value->ortApi->GetOnnxTypeFromTypeInfo(
      value->ortTypeInfo,
      out
    );
  }
}

-------------------------------------------------------------------------------
-- OrtApi::CreateTensorTypeAndShapeInfo

{-
> ORT_API2_STATUS(CreateTensorTypeAndShapeInfo,
>  _Outptr_ OrtTensorTypeAndShapeInfo** out
> );
-}

-- TODO: unimplemented

-------------------------------------------------------------------------------
-- OrtApi::SetTensorElementType

{-
> ORT_API2_STATUS(SetTensorElementType,
>  _Inout_ OrtTensorTypeAndShapeInfo* info,
>  enum ONNXTensorElementDataType type
> );
-}

-- TODO: unimplemented

-------------------------------------------------------------------------------
-- OrtApi::SetDimensions

{-
> ORT_API2_STATUS(SetDimensions,
>  OrtTensorTypeAndShapeInfo* info,
>  _In_ const int64_t* dim_values,
>  size_t dim_count
> );
-}

-- TODO: unimplemented

-------------------------------------------------------------------------------
-- OrtApi::GetTensorElementType

{-
> ORT_API2_STATUS(GetTensorElementType,
>  _In_ const OrtTensorTypeAndShapeInfo* info,
>  _Out_ enum ONNXTensorElementDataType* out
> );
-}
ortApiGetTensorElementType ::
  OrtTensorTypeAndShapeInfo ->
  IO ONNXTensorElementDataType
ortApiGetTensorElementType ortTensortTypeAndShapeInfo = do
  ortApi <- getOrtApi ortTensortTypeAndShapeInfo
  withOrtTensorTypeAndShapeInfoPtr ortTensortTypeAndShapeInfo $ \ortTensortTypeAndShapeInfoPtr ->
    alloca $ \outPtr -> do
      ortStatusPtr <-
        _wrap_OrtApi_GetTensorElementType
          (ConstPtr ortTensortTypeAndShapeInfoPtr)
          outPtr
      handleOrtStatus ortApi ortStatusPtr $
        ONNXTensorElementDataType <$>
          peek outPtr

foreign import capi unsafe
  "Onnxruntime/CApi_hsc.h _wrap_OrtApi_GetTensorElementType"
  _wrap_OrtApi_GetTensorElementType ::
    ConstPtr OrtTensorTypeAndShapeInfo ->
    Ptr ( #{type ONNXTensorElementDataType} ) ->
    IO (Ptr OrtStatus)

#{def
  OrtStatus* _wrap_OrtApi_GetTensorElementType(
    const HsOrtTensorTypeAndShapeInfo* value,
    ONNXTensorElementDataType* out
  ) {
    return value->ortApi->GetTensorElementType(
      value->ortTensorTypeAndShapeInfo,
      out
    );
  }
}

-------------------------------------------------------------------------------
-- OrtApi::GetDimensionsCount

{-
> ORT_API2_STATUS(GetDimensionsCount,
>  _In_ const OrtTensorTypeAndShapeInfo* info,
>  _Out_ size_t* out
> );
-}
ortApiGetDimensionsCount ::
  OrtTensorTypeAndShapeInfo ->
  IO Word64
ortApiGetDimensionsCount ortTensortTypeAndShapeInfo = do
  ortApi <- getOrtApi ortTensortTypeAndShapeInfo
  withOrtTensorTypeAndShapeInfoPtr ortTensortTypeAndShapeInfo $ \ortTensortTypeAndShapeInfoPtr ->
    alloca $ \outPtr -> do
      ortStatusPtr <-
        _wrap_OrtApi_GetDimensionsCount
          (ConstPtr ortTensortTypeAndShapeInfoPtr)
          outPtr
      handleOrtStatus ortApi ortStatusPtr $ do
        CSize dimValuesLen <- peek outPtr
        pure dimValuesLen

foreign import capi unsafe
  "Onnxruntime/CApi_hsc.h _wrap_OrtApi_GetDimensionsCount"
  _wrap_OrtApi_GetDimensionsCount ::
    ConstPtr OrtTensorTypeAndShapeInfo ->
    Ptr CSize ->
    IO (Ptr OrtStatus)

#{def
  OrtStatus* _wrap_OrtApi_GetDimensionsCount(
    const HsOrtTensorTypeAndShapeInfo* value,
    size_t* out
  ) {
    return value->ortApi->GetDimensionsCount(
      value->ortTensorTypeAndShapeInfo,
      out
    );
  }
}

-------------------------------------------------------------------------------
-- OrtApi::GetDimensions

{-
> ORT_API2_STATUS(GetDimensions,
>  _In_ const OrtTensorTypeAndShapeInfo* info,
>  _Out_ int64_t* dim_values,
>  size_t dim_values_length
> );
-}
ortApiGetDimensions ::
  OrtTensorTypeAndShapeInfo ->
  IO [Int64]
ortApiGetDimensions ortTensortTypeAndShapeInfo = do
  dimValuesLen <- ortApiGetDimensionsCount ortTensortTypeAndShapeInfo
  ortApi <- getOrtApi ortTensortTypeAndShapeInfo
  withOrtTensorTypeAndShapeInfoPtr ortTensortTypeAndShapeInfo $ \ortTensortTypeAndShapeInfoPtr ->
    alloca $ \dimValuesPtr -> do
      ortStatusPtr <-
        _wrap_OrtApi_GetDimensions
          (ConstPtr ortTensortTypeAndShapeInfoPtr)
          dimValuesPtr
          (CSize dimValuesLen)
      handleOrtStatus ortApi ortStatusPtr $
        peekArray (fromIntegral dimValuesLen) dimValuesPtr

foreign import capi unsafe
  "Onnxruntime/CApi_hsc.h _wrap_OrtApi_GetDimensions"
  _wrap_OrtApi_GetDimensions ::
    ConstPtr OrtTensorTypeAndShapeInfo ->
    Ptr ( #{type int64_t} ) ->
    CSize ->
    IO (Ptr OrtStatus)

#{def
  OrtStatus* _wrap_OrtApi_GetDimensions(
    const HsOrtTensorTypeAndShapeInfo* value,
    int64_t* dim_values,
    size_t dim_values_length
  ) {
    return value->ortApi->GetDimensions(
      value->ortTensorTypeAndShapeInfo,
      dim_values,
      dim_values_length
    );
  }
}

-------------------------------------------------------------------------------
-- OrtApi::GetSymbolicDimensions

{-
> ORT_API2_STATUS(GetSymbolicDimensions,
>  _In_ const OrtTensorTypeAndShapeInfo* info,
>  _Out_writes_all_(dim_params_length) const char* dim_params[],
>  size_t dim_params_length
> );
-}

-- TODO: unimplemented

-------------------------------------------------------------------------------
-- OrtApi::GetTensorShapeElementCount

{-
> ORT_API2_STATUS(GetTensorShapeElementCount,
>  _In_ const OrtTensorTypeAndShapeInfo* info,
>  _Out_ size_t* out
> );
-}
ortApiGetTensorShapeElementCount ::
  OrtTensorTypeAndShapeInfo ->
  IO Word64
ortApiGetTensorShapeElementCount ortTensortTypeAndShapeInfo = do
  ortApi <- getOrtApi ortTensortTypeAndShapeInfo
  withOrtTensorTypeAndShapeInfoPtr ortTensortTypeAndShapeInfo $ \ortTensortTypeAndShapeInfoPtr ->
    alloca $ \outPtr -> do
      ortStatusPtr <-
        _wrap_OrtApi_GetTensorShapeElementCount
          (ConstPtr ortTensortTypeAndShapeInfoPtr)
          outPtr
      handleOrtStatus ortApi ortStatusPtr $
        fromIntegral
          <$> peek outPtr

foreign import capi unsafe
  "Onnxruntime/CApi_hsc.h _wrap_OrtApi_GetTensorShapeElementCount"
  _wrap_OrtApi_GetTensorShapeElementCount ::
    ConstPtr OrtTensorTypeAndShapeInfo ->
    Ptr CSize ->
    IO (Ptr OrtStatus)

#{def
  OrtStatus* _wrap_OrtApi_GetTensorShapeElementCount(
    const HsOrtTensorTypeAndShapeInfo* value,
    size_t* out
  ) {
    return value->ortApi->GetTensorShapeElementCount(
      value->ortTensorTypeAndShapeInfo,
      out
    );
  }
}

-------------------------------------------------------------------------------
-- OrtApi::GetTensorTypeAndShape

{-
> ORT_API2_STATUS(GetTensorTypeAndShape,
>  _In_ const OrtValue* value,
>  _Outptr_ OrtTensorTypeAndShapeInfo** out
> );
-}
ortApiGetTensorTypeAndShape ::
  OrtValue ->
  IO OrtTensorTypeAndShapeInfo
ortApiGetTensorTypeAndShape ortValue = do
  ortApi <- getOrtApi ortValue
  withOrtValuePtr ortValue $ \ortValuePtr ->
    alloca $ \outPtr -> do
      ortStatusPtr <-
        _wrap_OrtApi_GetTensorTypeAndShape
          (ConstPtr ortValuePtr)
          outPtr
      handleOrtStatus ortApi ortStatusPtr $
        wrapCOrtTensorTypeAndShapeInfo ortApi
          =<< peek outPtr

foreign import capi unsafe
  "Onnxruntime/CApi_hsc.h _wrap_OrtApi_GetTensorTypeAndShape"
  _wrap_OrtApi_GetTensorTypeAndShape ::
    ConstPtr OrtValue ->
    Ptr (Ptr COrtTensorTypeAndShapeInfo) ->
    IO (Ptr OrtStatus)

#{def
  OrtStatus* _wrap_OrtApi_GetTensorTypeAndShape(
    const HsOrtValue* value,
    COrtTensorTypeAndShapeInfo** out
  ) {
    return value->ortApi->GetTensorTypeAndShape(
      value->ortValue,
      out
    );
  }
}

-------------------------------------------------------------------------------
-- OrtApi::GetTypeInfo

{-
> ORT_API2_STATUS(GetTypeInfo,
>  _In_ const OrtValue* value,
>  _Outptr_result_maybenull_ OrtTypeInfo** out
> );
-}
ortApiGetTypeInfo ::
  OrtValue ->
  IO OrtTypeInfo
ortApiGetTypeInfo ortValue = do
  ortApi <- getOrtApi ortValue
  withOrtValuePtr ortValue $ \ortValuePtr ->
    alloca $ \outPtr -> do
      ortStatusPtr <-
        _wrap_OrtApi_GetTypeInfo
          (ConstPtr ortValuePtr)
          outPtr
      handleOrtStatus ortApi ortStatusPtr $
        wrapCOrtTypeInfo ortApi
          =<< peek outPtr

foreign import capi unsafe
  "Onnxruntime/CApi_hsc.h _wrap_OrtApi_GetTypeInfo"
  _wrap_OrtApi_GetTypeInfo ::
    ConstPtr OrtValue ->
    Ptr (Ptr COrtTypeInfo) ->
    IO (Ptr OrtStatus)

#{def
  OrtStatus* _wrap_OrtApi_GetTypeInfo(
    const HsOrtValue* value,
    COrtTypeInfo** out
  ) {
    return value->ortApi->GetTypeInfo(
      value->ortValue,
      out
    );
  }
}

-------------------------------------------------------------------------------
-- OrtApi::GetValueType

{-
> ORT_API2_STATUS(GetValueType,
>  _In_ const OrtValue* value,
>  _Out_ enum ONNXType* out
> );
-}
ortApiGetValueType ::
  OrtValue ->
  IO ONNXType
ortApiGetValueType ortValue = do
  ortApi <- getOrtApi ortValue
  withOrtValuePtr ortValue $ \ortValuePtr ->
    alloca $ \outPtr -> do
      ortStatusPtr <-
        _wrap_OrtApi_GetValueType
          (ConstPtr ortValuePtr)
          outPtr
      handleOrtStatus ortApi ortStatusPtr $
        ONNXType
          <$> peek outPtr

foreign import capi unsafe
  "Onnxruntime/CApi_hsc.h _wrap_OrtApi_GetValueType"
  _wrap_OrtApi_GetValueType ::
    ConstPtr OrtValue ->
    Ptr ( #{type ONNXType} ) ->
    IO (Ptr OrtStatus)

#{def
  OrtStatus* _wrap_OrtApi_GetValueType(
    const HsOrtValue* value,
    ONNXType* out
  ) {
    return value->ortApi->GetValueType(
      value->ortValue,
      out
    );
  }
}

-------------------------------------------------------------------------------
-- OrtApi::CreateMemoryInfo

{-
> ORT_API2_STATUS(CreateMemoryInfo,
>  _In_ const char* name,
>  enum OrtAllocatorType type,
>  int id,
>  enum OrtMemType mem_type,
>  _Outptr_ OrtMemoryInfo** out
> );
-}
ortApiCreateMemoryInfo ::
  OrtApi ->
  String ->
  OrtAllocatorType ->
  Int ->
  OrtMemType ->
  IO OrtMemoryInfo
ortApiCreateMemoryInfo ortApi allocatorName allocatorType allocatorId memoryType = do
  withCString allocatorName $ \allocatorNamePtr -> do
    alloca $ \outPtr -> do
      ortStatusPtr <-
        _wrap_OrtApi_CreateMemoryInfo
          ortApi.ortApiConstPtr
          (ConstPtr allocatorNamePtr) -- NOTE: This is unsafe.
          allocatorType
          (fromIntegral allocatorId)
          memoryType
          outPtr
      handleOrtStatus ortApi ortStatusPtr $ do
        wrapCOrtMemoryInfo ortApi
          =<< peek outPtr

foreign import capi unsafe
  "Onnxruntime/CApi_hsc.h _wrap_OrtApi_CreateMemoryInfo"
  _wrap_OrtApi_CreateMemoryInfo ::
    ConstPtr OrtApi ->
    ConstPtr CChar ->
    OrtAllocatorType ->
    ( #{type int} ) ->
    OrtMemType ->
    Ptr (Ptr COrtMemoryInfo) ->
    IO (Ptr OrtStatus)

#{def
  OrtStatus* _wrap_OrtApi_CreateMemoryInfo(
    const OrtApi* ortApi,
    const char* name,
    enum OrtAllocatorType type,
    int id,
    enum OrtMemType mem_type,
    COrtMemoryInfo** out
  ) {
    return ortApi->CreateMemoryInfo(name, type, id, mem_type, out);
  }
}

-------------------------------------------------------------------------------
-- OrtApi::CreateCpuMemoryInfo

{-
> ORT_API2_STATUS(CreateCpuMemoryInfo,
>  enum OrtAllocatorType type,
>  enum OrtMemType mem_type,
>  _Outptr_ OrtMemoryInfo** out
> );
-}
ortApiCreateCpuMemoryInfo ::
  OrtApi ->
  OrtAllocatorType ->
  OrtMemType ->
  IO OrtMemoryInfo
ortApiCreateCpuMemoryInfo ortApi allocatorType memoryType = do
  alloca $ \outPtr -> do
    ortStatusPtr <-
      _wrap_OrtApi_CreateCpuMemoryInfo
        ortApi.ortApiConstPtr
        allocatorType
        memoryType
        outPtr
    handleOrtStatus ortApi ortStatusPtr $ do
      wrapCOrtMemoryInfo ortApi
        =<< peek outPtr

foreign import capi unsafe
  "Onnxruntime/CApi_hsc.h _wrap_OrtApi_CreateCpuMemoryInfo"
  _wrap_OrtApi_CreateCpuMemoryInfo ::
    ConstPtr OrtApi ->
    OrtAllocatorType ->
    OrtMemType ->
    Ptr (Ptr COrtMemoryInfo) ->
    IO (Ptr OrtStatus)

#{def
  OrtStatus* _wrap_OrtApi_CreateCpuMemoryInfo(
    const OrtApi* ortApi,
    enum OrtAllocatorType type,
    enum OrtMemType mem_type,
    COrtMemoryInfo** out
  ) {
    return ortApi->CreateCpuMemoryInfo(type, mem_type, out);
  }
}

-------------------------------------------------------------------------------
-- OrtApi::CompareMemoryInfo

{-
> ORT_API2_STATUS(CompareMemoryInfo,
>  _In_ const OrtMemoryInfo* info1,
>  _In_ const OrtMemoryInfo* info2,
>  _Out_ int* out
> );
-}

-- TODO: unimplemented

-------------------------------------------------------------------------------
-- OrtApi::MemoryInfoGetName

{-
> ORT_API2_STATUS(MemoryInfoGetName,
>  _In_ const OrtMemoryInfo* ptr,
>  _Out_ const char** out
> );
-}

-- TODO: unimplemented

-------------------------------------------------------------------------------
-- OrtApi::MemoryInfoGetId

{-
> ORT_API2_STATUS(MemoryInfoGetId,
>  _In_ const OrtMemoryInfo* ptr,
>  _Out_ int* out
> );
-}

-- TODO: unimplemented

-------------------------------------------------------------------------------
-- OrtApi::MemoryInfoGetMemType

{-
> ORT_API2_STATUS(MemoryInfoGetMemType,
>  _In_ const OrtMemoryInfo* ptr,
>  _Out_ OrtMemType* out
> );
-}

-- TODO: unimplemented

-------------------------------------------------------------------------------
-- OrtApi::MemoryInfoGetType

{-
> ORT_API2_STATUS(MemoryInfoGetType,
>  _In_ const OrtMemoryInfo* ptr,
>  _Out_ OrtAllocatorType* out
> );
-}

-- TODO: unimplemented

-------------------------------------------------------------------------------
-- OrtApi::AllocatorAlloc

{-
> ORT_API2_STATUS(AllocatorAlloc,
>  _Inout_ OrtAllocator* ort_allocator,
>  size_t size,
>  _Outptr_ void** out
> );
-}

-- TODO: unimplemented

-------------------------------------------------------------------------------
-- OrtApi::AllocatorFree

{- |
> ORT_API2_STATUS(AllocatorFree,
>  _Inout_ OrtAllocator* ort_allocator,
>  void* p
> );
-}
ortApiAllocatorFree ::
  OrtAllocator ->
  Ptr a ->
  IO ()
ortApiAllocatorFree allocator ptr = do
  ortApi <- getOrtApi allocator
  withOrtAllocatorPtr allocator $ \allocatorPtr -> do
    ortStatusPtr <-
      _wrap_OrtApi_AllocatorFree
        allocatorPtr
        ptr
    handleOrtStatus ortApi ortStatusPtr $ do
      pure ()

foreign import capi unsafe
  "Onnxruntime/CApi_hsc.h _wrap_OrtApi_AllocatorFree"
  _wrap_OrtApi_AllocatorFree ::
    Ptr OrtAllocator ->
    Ptr a ->
    IO (Ptr OrtStatus)

#{def
  OrtStatus* _wrap_OrtApi_AllocatorFree(
    HsOrtAllocator* allocator,
    void* p
  ) {
    return allocator->ortApi->AllocatorFree(
      allocator->ortAllocator,
      p
    );
  }
}

-------------------------------------------------------------------------------
-- OrtApi::AllocatorGetInfo

{-
> ORT_API2_STATUS(AllocatorGetInfo,
>  _In_ const OrtAllocator* ort_allocator,
>  _Outptr_ const struct OrtMemoryInfo** out
> );
-}

-- TODO: unimplemented

-------------------------------------------------------------------------------
-- OrtApi::GetAllocatorWithDefaultOptions

{- |
> ORT_API2_STATUS(GetAllocatorWithDefaultOptions,
>  _Outptr_ OrtAllocator** out
> );
-}
ortApiGetAllocatorWithDefaultOptions ::
  OrtApi ->
  IO OrtAllocator
ortApiGetAllocatorWithDefaultOptions ortApi = do
  alloca $ \outPtr -> do
    ortStatusPtr <-
      _wrap_OrtApi_GetAllocatorWithDefaultOptions
        ortApi.ortApiConstPtr
        outPtr
    handleOrtStatus ortApi ortStatusPtr $ do
      wrapCOrtAllocator ortApi
        =<< peek outPtr

foreign import capi unsafe
  "Onnxruntime/CApi_hsc.h _wrap_OrtApi_GetAllocatorWithDefaultOptions"
  _wrap_OrtApi_GetAllocatorWithDefaultOptions ::
    ConstPtr OrtApi ->
    Ptr (Ptr COrtAllocator) ->
    IO (Ptr OrtStatus)

#{def
  OrtStatus* _wrap_OrtApi_GetAllocatorWithDefaultOptions(
    const OrtApi* ortApi,
    OrtAllocator** out
  ) {
    return ortApi->GetAllocatorWithDefaultOptions(out);
  }
}

-------------------------------------------------------------------------------
-- OrtApi::AddFreeDimensionOverride

{- |
> ORT_API2_STATUS(AddFreeDimensionOverride,
>  _Inout_ OrtSessionOptions* options,
>  _In_ const char* dim_denotation,
>  _In_ int64_t dim_value
> );
-}
ortApiAddFreeDimensionOverride ::
  OrtSessionOptions ->
  String ->
  Int ->
  IO ()
ortApiAddFreeDimensionOverride options dimDenotation dimValue = do
  ortApi <- getOrtApi options
  withCString dimDenotation $ \dimDenotationPtr ->
    withOrtSessionOptionsPtr options $ \optionsPtr -> do
      ortStatusPtr <-
        _wrap_OrtApi_AddFreeDimensionOverride
          optionsPtr
          (ConstPtr dimDenotationPtr) -- NOTE: This is unsafe.
          (fromIntegral dimValue)
      handleOrtStatus ortApi ortStatusPtr $ do
        pure ()

foreign import capi unsafe
  "Onnxruntime/CApi_hsc.h _wrap_OrtApi_AddFreeDimensionOverride"
  _wrap_OrtApi_AddFreeDimensionOverride ::
    Ptr OrtSessionOptions ->
    ConstPtr CChar ->
    ( #{type int64_t} ) ->
    IO (Ptr OrtStatus)

#{def
  OrtStatus* _wrap_OrtApi_AddFreeDimensionOverride(
    HsOrtSessionOptions* options,
    const char* dimDenotation,
    int64_t dimValue
  ) {
    return options->ortApi->AddFreeDimensionOverride(
      options->ortSessionOptions,
      dimDenotation,
      dimValue
    );
  }
}

-------------------------------------------------------------------------------
-- OrtApi::GetValue

{-
> ORT_API2_STATUS(GetValue,
>   _In_ const OrtValue* value,
>   int index,
>   _Inout_ OrtAllocator* allocator,
>   _Outptr_ OrtValue** out
> );
-}

-- TODO: unimplemented

-------------------------------------------------------------------------------
-- OrtApi::GetValueCount

{-
> ORT_API2_STATUS(GetValueCount,
>   _In_ const OrtValue* value,
>   _Out_ size_t* out
> );
-}

-- TODO: unimplemented

-------------------------------------------------------------------------------
-- OrtApi::CreateValue

{-
> ORT_API2_STATUS(CreateValue,
>   _In_reads_(num_values) const OrtValue* const* in,
>   size_t num_values,
>   enum ONNXType value_type,
>   _Outptr_ OrtValue** out
> );
-}

-- TODO: unimplemented

-------------------------------------------------------------------------------
-- OrtApi::CreateOpaqueValue

{-
> ORT_API2_STATUS(CreateOpaqueValue,
>   _In_z_ const char* domain_name,
>   _In_z_ const char* type_name,
>   _In_ const void* data_container,
>   size_t data_container_size,
>   _Outptr_ OrtValue** out
> );
-}

-- TODO: unimplemented

-------------------------------------------------------------------------------
-- OrtApi::GetOpaqueValue

{-
> ORT_API2_STATUS(GetOpaqueValue,
>   _In_ const char* domain_name,
>   _In_ const char* type_name,
>   _In_ const OrtValue* in,
>   _Out_ void* data_container,
>   size_t data_container_size
> );
-}

-- TODO: unimplemented

-------------------------------------------------------------------------------
-- OrtApi::KernelInfoGetAttribute_float

-- ORT_API2_STATUS(KernelInfoGetAttribute_float, _In_ const OrtKernelInfo* info, _In_ const char* name, _Out_ float* out);

-- TODO: unimplemented

-------------------------------------------------------------------------------
-- OrtApi::KernelInfoGetAttribute_int64

-- ORT_API2_STATUS(KernelInfoGetAttribute_int64, _In_ const OrtKernelInfo* info, _In_ const char* name, _Out_ int64_t* out);

-- TODO: unimplemented

-------------------------------------------------------------------------------
-- OrtApi::KernelInfoGetAttribute_string

-- ORT_API2_STATUS(KernelInfoGetAttribute_string, _In_ const OrtKernelInfo* info, _In_ const char* name, _Out_ char* out, _Inout_ size_t* size);

-- TODO: unimplemented

-------------------------------------------------------------------------------
-- OrtApi::KernelContext_GetInputCount

-- ORT_API2_STATUS(KernelContext_GetInputCount, _In_ const OrtKernelContext* context, _Out_ size_t* out);

-- TODO: unimplemented

-------------------------------------------------------------------------------
-- OrtApi::KernelContext_GetOutputCount

-- ORT_API2_STATUS(KernelContext_GetOutputCount, _In_ const OrtKernelContext* context, _Out_ size_t* out);

-- TODO: unimplemented

-------------------------------------------------------------------------------
-- OrtApi::KernelContext_GetInput

-- ORT_API2_STATUS(KernelContext_GetInput, _In_ const OrtKernelContext* context, _In_ size_t index, _Out_ const OrtValue** out);

-- TODO: unimplemented

-------------------------------------------------------------------------------
-- OrtApi::KernelContext_GetOutput

-- ORT_API2_STATUS(KernelContext_GetOutput, _Inout_ OrtKernelContext* context, _In_ size_t index, _In_ const int64_t* dim_values, size_t dim_count, _Outptr_ OrtValue** out);

-- TODO: unimplemented

-------------------------------------------------------------------------------
-- OrtApi::GetDenotationFromTypeInfo

{-
> ORT_API2_STATUS(GetDenotationFromTypeInfo,
>   _In_ const OrtTypeInfo* type_info,
>   _Out_ const char** const denotation,
>   _Out_ size_t* len
> );
-}

-- TODO: unimplemented

-------------------------------------------------------------------------------
-- OrtApi::CastTypeInfoToMapTypeInfo

{-
> ORT_API2_STATUS(CastTypeInfoToMapTypeInfo,
>   _In_ const OrtTypeInfo* type_info,
>   _Outptr_result_maybenull_ const OrtMapTypeInfo** out
> );
-}

-- TODO: unimplemented

-------------------------------------------------------------------------------
-- OrtApi::CastTypeInfoToSequenceTypeInfo

{-
> ORT_API2_STATUS(CastTypeInfoToSequenceTypeInfo,
>   _In_ const OrtTypeInfo* type_info,
>   _Outptr_result_maybenull_ const OrtSequenceTypeInfo** out
> );
-}

-- TODO: unimplemented


--------------------------------------------------------------------------------
-- OrtApi::GetMapKeyType

-- > ORT_API2_STATUS(GetMapKeyType, _In_ const OrtMapTypeInfo* map_type_info, _Out_ enum ONNXTensorElementDataType* out);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::GetMapValueType

-- > ORT_API2_STATUS(GetMapValueType, _In_ const OrtMapTypeInfo* map_type_info, _Outptr_ OrtTypeInfo** type_info);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::GetSequenceElementType

-- > ORT_API2_STATUS(GetSequenceElementType, _In_ const OrtSequenceTypeInfo* sequence_type_info, _Outptr_ OrtTypeInfo** type_info);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::SessionEndProfiling

-- > ORT_API2_STATUS(SessionEndProfiling, _In_ OrtSession* session, _Inout_ OrtAllocator* allocator, _Outptr_ char** out);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::SessionGetModelMetadata

-- > ORT_API2_STATUS(SessionGetModelMetadata, _In_ const OrtSession* session, _Outptr_ OrtModelMetadata** out);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::ModelMetadataGetProducerName

-- > ORT_API2_STATUS(ModelMetadataGetProducerName, _In_ const OrtModelMetadata* model_metadata, _Inout_ OrtAllocator* allocator, _Outptr_ char** value);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::ModelMetadataGetGraphName

-- > ORT_API2_STATUS(ModelMetadataGetGraphName, _In_ const OrtModelMetadata* model_metadata, _Inout_ OrtAllocator* allocator, _Outptr_ char** value);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::ModelMetadataGetDomain

-- > ORT_API2_STATUS(ModelMetadataGetDomain, _In_ const OrtModelMetadata* model_metadata, _Inout_ OrtAllocator* allocator, _Outptr_ char** value);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::ModelMetadataGetDescription

-- > ORT_API2_STATUS(ModelMetadataGetDescription, _In_ const OrtModelMetadata* model_metadata, _Inout_ OrtAllocator* allocator, _Outptr_ char** value);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::ModelMetadataLookupCustomMetadataMap

-- > ORT_API2_STATUS(ModelMetadataLookupCustomMetadataMap, _In_ const OrtModelMetadata* model_metadata, _Inout_ OrtAllocator* allocator, _In_ const char* key, _Outptr_result_maybenull_ char** value);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::ModelMetadataGetVersion

-- > ORT_API2_STATUS(ModelMetadataGetVersion, _In_ const OrtModelMetadata* model_metadata, _Out_ int64_t* value);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::CreateEnvWithGlobalThreadPools

-- > ORT_API2_STATUS(CreateEnvWithGlobalThreadPools, OrtLoggingLevel log_severity_level, _In_ const char* logid, _In_ const OrtThreadingOptions* tp_options, _Outptr_ OrtEnv** out);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::DisablePerSessionThreads

-- > ORT_API2_STATUS(DisablePerSessionThreads, _Inout_ OrtSessionOptions* options);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::CreateThreadingOptions

-- > ORT_API2_STATUS(CreateThreadingOptions, _Outptr_ OrtThreadingOptions** out);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::ModelMetadataGetCustomMetadataMapKeys

-- > ORT_API2_STATUS(ModelMetadataGetCustomMetadataMapKeys, _In_ const OrtModelMetadata* model_metadata, _Inout_ OrtAllocator* allocator, _Outptr_result_buffer_maybenull_(*num_keys) char*** keys, _Out_ int64_t* num_keys);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::AddFreeDimensionOverrideByName

-- > ORT_API2_STATUS(AddFreeDimensionOverrideByName, _Inout_ OrtSessionOptions* options, _In_ const char* dim_name, _In_ int64_t dim_value);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::GetAvailableProviders

-- > ORT_API2_STATUS(GetAvailableProviders, _Outptr_ char*** out_ptr, _Out_ int* provider_length);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::ReleaseAvailableProviders

-- > ORT_API2_STATUS(ReleaseAvailableProviders, _In_ char** ptr, _In_ int providers_length);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::GetStringTensorElementLength

-- > ORT_API2_STATUS(GetStringTensorElementLength, _In_ const OrtValue* value, size_t index, _Out_ size_t* out);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::GetStringTensorElement

-- > ORT_API2_STATUS(GetStringTensorElement, _In_ const OrtValue* value, size_t s_len, size_t index, _Out_writes_bytes_all_(s_len) void* s);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::FillStringTensorElement

-- > ORT_API2_STATUS(FillStringTensorElement, _Inout_ OrtValue* value, _In_ const char* s, size_t index);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::AddSessionConfigEntry

-- > ORT_API2_STATUS(AddSessionConfigEntry, _Inout_ OrtSessionOptions* options, _In_z_ const char* config_key, _In_z_ const char* config_value);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::CreateAllocator

-- > ORT_API2_STATUS(CreateAllocator, _In_ const OrtSession* session, _In_ const OrtMemoryInfo* mem_info, _Outptr_ OrtAllocator** out);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::RunWithBinding

-- > ORT_API2_STATUS(RunWithBinding, _Inout_ OrtSession* session, _In_ const OrtRunOptions* run_options, _In_ const OrtIoBinding* binding_ptr);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::CreateIoBinding

-- > ORT_API2_STATUS(CreateIoBinding, _Inout_ OrtSession* session, _Outptr_ OrtIoBinding** out);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::BindInput

-- > ORT_API2_STATUS(BindInput, _Inout_ OrtIoBinding* binding_ptr, _In_ const char* name, _In_ const OrtValue* val_ptr);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::BindOutput

-- > ORT_API2_STATUS(BindOutput, _Inout_ OrtIoBinding* binding_ptr, _In_ const char* name, _In_ const OrtValue* val_ptr);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::BindOutputToDevice

-- > ORT_API2_STATUS(BindOutputToDevice, _Inout_ OrtIoBinding* binding_ptr, _In_ const char* name, _In_ const OrtMemoryInfo* mem_info_ptr);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::GetBoundOutputNames

-- > ORT_API2_STATUS(GetBoundOutputNames, _In_ const OrtIoBinding* binding_ptr, _In_ OrtAllocator* allocator, _Out_ char** buffer, _Out_writes_all_(count) size_t** lengths, _Out_ size_t* count);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::GetBoundOutputValues

-- > ORT_API2_STATUS(GetBoundOutputValues, _In_ const OrtIoBinding* binding_ptr, _In_ OrtAllocator* allocator, _Out_writes_all_(output_count) OrtValue*** output, _Out_ size_t* output_count);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::ClearBoundInputs

-- > void(ORT_API_CALL* ClearBoundInputs)(_Inout_ OrtIoBinding* binding_ptr) NO_EXCEPTION ORT_ALL_ARGS_NONNULL;

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::ClearBoundOutputs

-- > void(ORT_API_CALL* ClearBoundOutputs)(_Inout_ OrtIoBinding* binding_ptr) NO_EXCEPTION ORT_ALL_ARGS_NONNULL;

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::TensorAt

-- > ORT_API2_STATUS(TensorAt, _Inout_ OrtValue* value, const int64_t* location_values, size_t location_values_count, _Outptr_ void** out);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::CreateAndRegisterAllocator

-- > ORT_API2_STATUS(CreateAndRegisterAllocator, _Inout_ OrtEnv* env, _In_ const OrtMemoryInfo* mem_info, _In_ const OrtArenaCfg* arena_cfg);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::SetLanguageProjection

-- > ORT_API2_STATUS(SetLanguageProjection, _In_ const OrtEnv* ort_env, _In_ OrtLanguageProjection projection);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::SessionGetProfilingStartTimeNs

-- > ORT_API2_STATUS(SessionGetProfilingStartTimeNs, _In_ const OrtSession* session, _Outptr_ uint64_t* out);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::SetGlobalIntraOpNumThreads

-- > ORT_API2_STATUS(SetGlobalIntraOpNumThreads, _Inout_ OrtThreadingOptions* tp_options, int intra_op_num_threads);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::SetGlobalInterOpNumThreads

-- > ORT_API2_STATUS(SetGlobalInterOpNumThreads, _Inout_ OrtThreadingOptions* tp_options, int inter_op_num_threads);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::SetGlobalSpinControl

-- > ORT_API2_STATUS(SetGlobalSpinControl, _Inout_ OrtThreadingOptions* tp_options, int allow_spinning);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::AddInitializer

-- > ORT_API2_STATUS(AddInitializer, _Inout_ OrtSessionOptions* options, _In_z_ const char* name, _In_ const OrtValue* val);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::CreateEnvWithCustomLoggerAndGlobalThreadPools

-- > ORT_API2_STATUS(CreateEnvWithCustomLoggerAndGlobalThreadPools, OrtLoggingFunction logging_function, _In_opt_ void* logger_param, OrtLoggingLevel log_severity_level, _In_ const char* logid, _In_ const struct OrtThreadingOptions* tp_options, _Outptr_ OrtEnv** out);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::SessionOptionsAppendExecutionProvider_CUDA

-- > ORT_API2_STATUS(SessionOptionsAppendExecutionProvider_CUDA, _In_ OrtSessionOptions* options, _In_ const OrtCUDAProviderOptions* cuda_options);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::SessionOptionsAppendExecutionProvider_ROCM

-- > ORT_API2_STATUS(SessionOptionsAppendExecutionProvider_ROCM, _In_ OrtSessionOptions* options, _In_ const OrtROCMProviderOptions* rocm_options);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::SessionOptionsAppendExecutionProvider_OpenVINO

-- > ORT_API2_STATUS(SessionOptionsAppendExecutionProvider_OpenVINO, _In_ OrtSessionOptions* options, _In_ const OrtOpenVINOProviderOptions* provider_options);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::SetGlobalDenormalAsZero

-- > ORT_API2_STATUS(SetGlobalDenormalAsZero, _Inout_ OrtThreadingOptions* tp_options);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::CreateArenaCfg

-- > ORT_API2_STATUS(CreateArenaCfg, _In_ size_t max_mem, int arena_extend_strategy, int initial_chunk_size_bytes, int max_dead_bytes_per_chunk, _Outptr_ OrtArenaCfg** out);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::ModelMetadataGetGraphDescription

-- > ORT_API2_STATUS(ModelMetadataGetGraphDescription, _In_ const OrtModelMetadata* model_metadata, _Inout_ OrtAllocator* allocator, _Outptr_ char** value);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::SessionOptionsAppendExecutionProvider_TensorRT

-- > ORT_API2_STATUS(SessionOptionsAppendExecutionProvider_TensorRT, _In_ OrtSessionOptions* options, _In_ const OrtTensorRTProviderOptions* tensorrt_options);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::SetCurrentGpuDeviceId

-- > ORT_API2_STATUS(SetCurrentGpuDeviceId, _In_ int device_id);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::GetCurrentGpuDeviceId

-- > ORT_API2_STATUS(GetCurrentGpuDeviceId, _In_ int* device_id);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::KernelInfoGetAttributeArray_float

-- > ORT_API2_STATUS(KernelInfoGetAttributeArray_float, _In_ const OrtKernelInfo* info, _In_ const char* name, _Out_ float* out, _Inout_ size_t* size);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::KernelInfoGetAttributeArray_int64

-- > ORT_API2_STATUS(KernelInfoGetAttributeArray_int64, _In_ const OrtKernelInfo* info, _In_ const char* name, _Out_ int64_t* out, _Inout_ size_t* size);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::CreateArenaCfgV2

-- > ORT_API2_STATUS(CreateArenaCfgV2, _In_reads_(num_keys) const char* const* arena_config_keys, _In_reads_(num_keys) const size_t* arena_config_values, _In_ size_t num_keys, _Outptr_ OrtArenaCfg** out);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::AddRunConfigEntry

-- > ORT_API2_STATUS(AddRunConfigEntry, _Inout_ OrtRunOptions* options, _In_z_ const char* config_key, _In_z_ const char* config_value);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::CreatePrepackedWeightsContainer

-- > ORT_API2_STATUS(CreatePrepackedWeightsContainer, _Outptr_ OrtPrepackedWeightsContainer** out);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::CreateSessionWithPrepackedWeightsContainer

-- > ORT_API2_STATUS(CreateSessionWithPrepackedWeightsContainer, _In_ const OrtEnv* env, _In_ const ORTCHAR_T* model_path, _In_ const OrtSessionOptions* options, _Inout_ OrtPrepackedWeightsContainer* prepacked_weights_container, _Outptr_ OrtSession** out);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::CreateSessionFromArrayWithPrepackedWeightsContainer

-- > ORT_API2_STATUS(CreateSessionFromArrayWithPrepackedWeightsContainer, _In_ const OrtEnv* env, _In_ const void* model_data, size_t model_data_length, _In_ const OrtSessionOptions* options, _Inout_ OrtPrepackedWeightsContainer* prepacked_weights_container, _Outptr_ OrtSession** out);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::SessionOptionsAppendExecutionProvider_TensorRT_V2

-- > ORT_API2_STATUS(SessionOptionsAppendExecutionProvider_TensorRT_V2, _In_ OrtSessionOptions* options, _In_ const OrtTensorRTProviderOptionsV2* tensorrt_options);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::CreateTensorRTProviderOptions

-- > ORT_API2_STATUS(CreateTensorRTProviderOptions, _Outptr_ OrtTensorRTProviderOptionsV2** out);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::UpdateTensorRTProviderOptions

-- > ORT_API2_STATUS(UpdateTensorRTProviderOptions, _Inout_ OrtTensorRTProviderOptionsV2* tensorrt_options, _In_reads_(num_keys) const char* const* provider_options_keys, _In_reads_(num_keys) const char* const* provider_options_values, _In_ size_t num_keys);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::GetTensorRTProviderOptionsAsString

-- > ORT_API2_STATUS(GetTensorRTProviderOptionsAsString, _In_ const OrtTensorRTProviderOptionsV2* tensorrt_options, _Inout_ OrtAllocator* allocator, _Outptr_ char** ptr);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::ReleaseTensorRTProviderOptions

-- > void(ORT_API_CALL* ReleaseTensorRTProviderOptions)(_Frees_ptr_opt_ OrtTensorRTProviderOptionsV2* input);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::EnableOrtCustomOps

-- > ORT_API2_STATUS(EnableOrtCustomOps, _Inout_ OrtSessionOptions* options);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::RegisterAllocator

-- > ORT_API2_STATUS(RegisterAllocator, _Inout_ OrtEnv* env, _In_ OrtAllocator* allocator);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::UnregisterAllocator

-- > ORT_API2_STATUS(UnregisterAllocator, _Inout_ OrtEnv* env, _In_ const OrtMemoryInfo* mem_info);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::IsSparseTensor

-- > ORT_API2_STATUS(IsSparseTensor, _In_ const OrtValue* value, _Out_ int* out);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::CreateSparseTensorAsOrtValue

-- > ORT_API2_STATUS(CreateSparseTensorAsOrtValue, _Inout_ OrtAllocator* allocator, _In_ const int64_t* dense_shape, size_t dense_shape_len, ONNXTensorElementDataType type, _Outptr_ OrtValue** out);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::FillSparseTensorCoo

-- > ORT_API2_STATUS(FillSparseTensorCoo, _Inout_ OrtValue* ort_value, _In_ const OrtMemoryInfo* data_mem_info, _In_ const int64_t* values_shape, size_t values_shape_len, _In_ const void* values, _In_ const int64_t* indices_data, size_t indices_num);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::FillSparseTensorCsr

-- > ORT_API2_STATUS(FillSparseTensorCsr, _Inout_ OrtValue* ort_value, _In_ const OrtMemoryInfo* data_mem_info, _In_ const int64_t* values_shape, size_t values_shape_len, _In_ const void* values, _In_ const int64_t* inner_indices_data, size_t inner_indices_num, _In_ const int64_t* outer_indices_data, size_t outer_indices_num);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::FillSparseTensorBlockSparse

-- > ORT_API2_STATUS(FillSparseTensorBlockSparse, _Inout_ OrtValue* ort_value, _In_ const OrtMemoryInfo* data_mem_info, _In_ const int64_t* values_shape, size_t values_shape_len, _In_ const void* values, _In_ const int64_t* indices_shape_data, size_t indices_shape_len, _In_ const int32_t* indices_data);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::CreateSparseTensorWithValuesAsOrtValue

-- > ORT_API2_STATUS(CreateSparseTensorWithValuesAsOrtValue, _In_ const OrtMemoryInfo* info, _Inout_ void* p_data, _In_ const int64_t* dense_shape, size_t dense_shape_len, _In_ const int64_t* values_shape, size_t values_shape_len, ONNXTensorElementDataType type, _Outptr_ OrtValue** out);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::UseCooIndices

-- > ORT_API2_STATUS(UseCooIndices, _Inout_ OrtValue* ort_value, _Inout_ int64_t* indices_data, size_t indices_num);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::UseCsrIndices

-- > ORT_API2_STATUS(UseCsrIndices, _Inout_ OrtValue* ort_value, _Inout_ int64_t* inner_data, size_t inner_num, _Inout_ int64_t* outer_data, size_t outer_num);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::UseBlockSparseIndices

-- > ORT_API2_STATUS(UseBlockSparseIndices, _Inout_ OrtValue* ort_value, const int64_t* indices_shape, size_t indices_shape_len, _Inout_ int32_t* indices_data);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::GetSparseTensorFormat

-- > ORT_API2_STATUS(GetSparseTensorFormat, _In_ const OrtValue* ort_value, _Out_ enum OrtSparseFormat* out);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::GetSparseTensorValuesTypeAndShape

-- > ORT_API2_STATUS(GetSparseTensorValuesTypeAndShape, _In_ const OrtValue* ort_value, _Outptr_ OrtTensorTypeAndShapeInfo** out);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::GetSparseTensorValues

-- > ORT_API2_STATUS(GetSparseTensorValues, _In_ const OrtValue* ort_value, _Outptr_ const void** out);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::GetSparseTensorIndicesTypeShape

-- > ORT_API2_STATUS(GetSparseTensorIndicesTypeShape, _In_ const OrtValue* ort_value, enum OrtSparseIndicesFormat indices_format, _Outptr_ OrtTensorTypeAndShapeInfo** out);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::GetSparseTensorIndices

-- > ORT_API2_STATUS(GetSparseTensorIndices, _In_ const OrtValue* ort_value, enum OrtSparseIndicesFormat indices_format, _Out_ size_t* num_indices, _Outptr_ const void** indices);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::HasValue

-- > ORT_API2_STATUS(HasValue, _In_ const OrtValue* value, _Out_ int* out);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::KernelContext_GetGPUComputeStream

-- > ORT_API2_STATUS(KernelContext_GetGPUComputeStream, _In_ const OrtKernelContext* context, _Outptr_ void** out);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::GetTensorMemoryInfo

-- > ORT_API2_STATUS(GetTensorMemoryInfo, _In_ const OrtValue* value, _Out_ const OrtMemoryInfo** mem_info);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::GetExecutionProviderApi

-- > ORT_API2_STATUS(GetExecutionProviderApi, _In_ const char* provider_name, _In_ uint32_t version, _Outptr_ const void** provider_api);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::SessionOptionsSetCustomCreateThreadFn

-- > ORT_API2_STATUS(SessionOptionsSetCustomCreateThreadFn, _Inout_ OrtSessionOptions* options, _In_ OrtCustomCreateThreadFn ort_custom_create_thread_fn);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::SessionOptionsSetCustomThreadCreationOptions

-- > ORT_API2_STATUS(SessionOptionsSetCustomThreadCreationOptions, _Inout_ OrtSessionOptions* options, _In_ void* ort_custom_thread_creation_options);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::SessionOptionsSetCustomJoinThreadFn

-- > ORT_API2_STATUS(SessionOptionsSetCustomJoinThreadFn, _Inout_ OrtSessionOptions* options, _In_ OrtCustomJoinThreadFn ort_custom_join_thread_fn);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::SetGlobalCustomCreateThreadFn

-- > ORT_API2_STATUS(SetGlobalCustomCreateThreadFn, _Inout_ OrtThreadingOptions* tp_options, _In_ OrtCustomCreateThreadFn ort_custom_create_thread_fn);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::SetGlobalCustomThreadCreationOptions

-- > ORT_API2_STATUS(SetGlobalCustomThreadCreationOptions, _Inout_ OrtThreadingOptions* tp_options, _In_ void* ort_custom_thread_creation_options);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::SetGlobalCustomJoinThreadFn

-- > ORT_API2_STATUS(SetGlobalCustomJoinThreadFn, _Inout_ OrtThreadingOptions* tp_options, _In_ OrtCustomJoinThreadFn ort_custom_join_thread_fn);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::SynchronizeBoundInputs

-- > ORT_API2_STATUS(SynchronizeBoundInputs, _Inout_ OrtIoBinding* binding_ptr);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::SynchronizeBoundOutputs

-- > ORT_API2_STATUS(SynchronizeBoundOutputs, _Inout_ OrtIoBinding* binding_ptr);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::SessionOptionsAppendExecutionProvider_CUDA_V2

-- > ORT_API2_STATUS(SessionOptionsAppendExecutionProvider_CUDA_V2, _In_ OrtSessionOptions* options, _In_ const OrtCUDAProviderOptionsV2* cuda_options);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::CreateCUDAProviderOptions

-- > ORT_API2_STATUS(CreateCUDAProviderOptions, _Outptr_ OrtCUDAProviderOptionsV2** out);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::UpdateCUDAProviderOptions

-- > ORT_API2_STATUS(UpdateCUDAProviderOptions, _Inout_ OrtCUDAProviderOptionsV2* cuda_options, _In_reads_(num_keys) const char* const* provider_options_keys, _In_reads_(num_keys) const char* const* provider_options_values, _In_ size_t num_keys);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::GetCUDAProviderOptionsAsString

-- > ORT_API2_STATUS(GetCUDAProviderOptionsAsString, _In_ const OrtCUDAProviderOptionsV2* cuda_options, _Inout_ OrtAllocator* allocator, _Outptr_ char** ptr);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::ReleaseCUDAProviderOptions

-- > void(ORT_API_CALL* ReleaseCUDAProviderOptions)(_Frees_ptr_opt_ OrtCUDAProviderOptionsV2* input);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::SessionOptionsAppendExecutionProvider_MIGraphX

-- > ORT_API2_STATUS(SessionOptionsAppendExecutionProvider_MIGraphX, _In_ OrtSessionOptions* options, _In_ const OrtMIGraphXProviderOptions* migraphx_options);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::AddExternalInitializers

-- > ORT_API2_STATUS(AddExternalInitializers, _In_ OrtSessionOptions* options, _In_reads_(num_initializers) const char* const* initializer_names, _In_reads_(num_initializers) const OrtValue* const* initializers, size_t num_initializers);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::CreateOpAttr

-- > ORT_API2_STATUS(CreateOpAttr, _In_ const char* name, _In_ const void* data, _In_ int len, _In_ OrtOpAttrType type, _Outptr_ OrtOpAttr** op_attr);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::CreateOp

-- > ORT_API2_STATUS(CreateOp, _In_ const OrtKernelInfo* info, _In_z_ const char* op_name, _In_z_ const char* domain, int version, _In_reads_(type_constraint_count) const char** type_constraint_names, _In_reads_(type_constraint_count) const ONNXTensorElementDataType* type_constraint_values, int type_constraint_count, _In_reads_(attr_count) const OrtOpAttr* const* attr_values, int attr_count, int input_count, int output_count, _Outptr_ OrtOp** ort_op);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::InvokeOp

-- > ORT_API2_STATUS(InvokeOp, _In_ const OrtKernelContext* context, _In_ const OrtOp* ort_op, _In_ const OrtValue* const* input_values, _In_ int input_count, _Inout_ OrtValue* const* output_values, _In_ int output_count);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::SessionOptionsAppendExecutionProvider

-- > ORT_API2_STATUS(SessionOptionsAppendExecutionProvider, _In_ OrtSessionOptions* options, _In_ const char* provider_name, _In_reads_(num_keys) const char* const* provider_options_keys, _In_reads_(num_keys) const char* const* provider_options_values, _In_ size_t num_keys);

-- TODO: unimplemented

--------------------------------------------------------------------------------
-- OrtApi::CopyKernelInfo

-- > ORT_API2_STATUS(CopyKernelInfo, _In_ const OrtKernelInfo* info, _Outptr_ OrtKernelInfo** info_copy);

-- TODO: unimplemented

-- ... and many more
-- ... genuinely, this is only about halfway through the header file
