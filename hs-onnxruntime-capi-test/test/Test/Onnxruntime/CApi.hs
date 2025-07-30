{-# OPTIONS_GHC -Wno-incomplete-uni-patterns #-}

module Test.Onnxruntime.CApi where

import Data.Foldable (for_)
import Data.List qualified as L
import Data.Vector.Storable (Vector)
import Data.Vector.Storable qualified as VS
import Data.Version (Version, makeVersion, parseVersion)
import Onnxruntime.CApi
import Test.Tasty (TestTree, testGroup)
import Test.Tasty.HUnit (assertBool, assertFailure, testCase, (@?), (@?=))
import Text.ParserCombinators.ReadP (readP_to_S)

tests :: TestTree
tests =
    testGroup
        "CApi"
        [ test_ortApiBaseGetVersionString
        , test_ortApiGetModelTypeInfo
        , test_ortApiRun
        ]

test_ortApiBaseGetVersionString :: TestTree
test_ortApiBaseGetVersionString =
    testCase "test_ortApiBaseGetVersionString" $ do
        ortApiBase <- ortGetApiBase
        versionString <- ortApiBaseGetVersionString ortApiBase
        case readVersion versionString of
            Nothing -> assertFailure $ "Could not parse version " <> versionString
            Just version -> version >= makeVersion [1, 21] @? "Onnxruntime version is >=1.21"

test_ortApiGetModelTypeInfo :: TestTree
test_ortApiGetModelTypeInfo = do
    testCase "test_ortApiGetModelTypeInfo" $ do
        -- Get OrtApi
        ortApiBase <- ortGetApiBase
        ortApi <- ortApiBaseGetApi ortApiBase ortApiVersion
        allocator <- ortApiGetAllocatorWithDefaultOptions ortApi
        -- Create OrtEnv
        let logid = "test_ortApiRun"
        ortEnv <- ortApiCreateEnv ortApi OrtLoggingLevelFatal logid
        -- Create OrtSessionOptions
        ortSessionOptions <- ortApiCreateSessionOptions ortApi
        ortApiSetSessionLogSeverityLevel ortSessionOptions OrtLoggingLevelFatal
        -- Create OrtSession
        -- TODO: make model path more robust
        let modelPath = "test/data/controller.onnx"
        ortSession <- ortApiCreateSession ortEnv modelPath ortSessionOptions
        -- Get the model input type and shape info
        inputCount <- ortApiSessionGetInputCount ortSession
        inputCount @?= 1
        for_ [0 .. inputCount - 1] $ \inputIndex -> do
            inputTypeInfo <- ortApiSessionGetInputTypeInfo ortSession inputIndex
            inputType <- ortApiGetOnnxTypeFromTypeInfo inputTypeInfo
            inputType @?= ONNXTypeTensor
            inputTypeAndShape <- ortApiCastTypeInfoToTensorInfo inputTypeInfo
            inputDims <- ortApiGetDimensions inputTypeAndShape
            inputDims @?= [1, 2]
            inputTensorElementType <- ortApiGetTensorElementType inputTypeAndShape
            inputTensorElementType @?= ONNXTensorElementDataTypeFloat
            inputName <- ortApiSessionGetInputName ortSession inputIndex allocator
            inputName @?= "input_1"
        -- Get the model output type and shape info
        outputCount <- ortApiSessionGetOutputCount ortSession
        outputCount @?= 1
        for_ [0 .. outputCount - 1] $ \outputIndex -> do
            outputTypeInfo <- ortApiSessionGetOutputTypeInfo ortSession outputIndex
            outputType <- ortApiGetOnnxTypeFromTypeInfo outputTypeInfo
            outputType @?= ONNXTypeTensor
            outputTypeAndShape <- ortApiCastTypeInfoToTensorInfo outputTypeInfo
            outputDims <- ortApiGetDimensions outputTypeAndShape
            outputDims @?= [1, 1]
            outputTensorElementType <- ortApiGetTensorElementType outputTypeAndShape
            outputTensorElementType @?= ONNXTensorElementDataTypeFloat
            outputName <- ortApiSessionGetOutputName ortSession outputIndex allocator
            outputName @?= "dense_3"

test_ortApiRun :: TestTree
test_ortApiRun = do
    testCase "test_ortApiRun" $ do
        -- Get OrtApi
        ortApiBase <- ortGetApiBase
        ortApi <- ortApiBaseGetApi ortApiBase ortApiVersion
        -- Create OrtEnv
        let logid = "test_ortApiRun"
        ortEnv <- ortApiCreateEnv ortApi OrtLoggingLevelFatal logid
        -- Create OrtSessionOptions
        ortSessionOptions <- ortApiCreateSessionOptions ortApi
        ortApiSetSessionLogSeverityLevel ortSessionOptions OrtLoggingLevelFatal
        -- Create OrtSession
        -- TODO: make model path more robust
        let modelPath = "test/data/controller.onnx"
        ortSession <- ortApiCreateSession ortEnv modelPath ortSessionOptions
        -- Create OrtRunOptions
        ortRunOptions <- ortApiCreateRunOptions ortApi
        -- Create input OrtValue
        let input1Data = VS.fromList [10, 10] :: Vector Float
        ortApiWithTensorWithDataAsOrtValue ortApi input1Data [1, 2] $ \input1 -> do
            -- Run
            let inputNames = ["input_1"]
            let outputNames = ["dense_3"]
            outputs <- ortApiRun ortSession ortRunOptions inputNames [input1] outputNames
            assertBool "there is one output" (length outputs == 1)
            let [output] = outputs
            outputIsTensor <- ortApiIsTensor output
            assertBool "the output is tensor" outputIsTensor
            outputTypeAndShape <- ortApiGetTensorTypeAndShape output
            outputDimensions <- ortApiGetDimensions outputTypeAndShape
            outputDimensions @?= [1, 1]
            outputElementCount <- ortApiGetTensorShapeElementCount outputTypeAndShape
            outputElementCount @?= 1
            ortApiWithTensorData output $ \(outputData :: Vector Float) -> do
                VS.length outputData @?= 1
                outputData VS.! 0 @?= -76

-- | Internal helper. Read a 'Version' from a 'String'.
readVersion :: String -> Maybe Version
readVersion = fmap fst . L.find (null . snd) . readP_to_S parseVersion
