open System
open System.IO
open Microsoft.ML
open Microsoft.ML.Data
open Microsoft.ML.Transforms

/// The Digit class represents one mnist digit.
[<CLIMutable>]
type Digit = {
    [<LoadColumn(0)>] Number : float32
    [<LoadColumn(1, 784)>] [<VectorType(784)>] PixelValues : float32[]
}

/// The DigitPrediction class represents one digit prediction.
[<CLIMutable>]
type DigitPrediction = {
    Score : float32[]
}

/// file paths to train and test data files (assumes os = windows!)
let trDataPath = sprintf "%s\\data\\mnist_train.csv" Environment.CurrentDirectory
let teDataPath = sprintf "%s\\data\\mnist_test.csv" Environment.CurrentDirectory

[<EntryPoint>]
let main argv = 

    // create a machine learning context
    let mlcontext = new MLContext()

    // load the datafiles
    let trData = mlcontext.Data.LoadFromTextFile<Digit>(trDataPath, hasHeader = true, separatorChar = ',')
    let teData = mlcontext.Data.LoadFromTextFile<Digit>(teDataPath, hasHeader = true, separatorChar = ',')

    // build a training pipeline
    let pipeline = 
        EstimatorChain()
            .Append(mlcontext.Transforms.Conversion.MapValueToKey("Label", "Number", keyOrdinality = ValueToKeyMappingEstimator.KeyOrdinality.ByValue))
            .Append(mlcontext.Transforms.Concatenate("Features", "PixelValues"))
            
            //cache data to speed up training                
            .AppendCacheCheckpoint(mlcontext)

            //train the model with SDCA
            .Append(mlcontext.MulticlassClassification.Trainers.SdcaMaximumEntropy())
            
            .Append(mlcontext.Transforms.Conversion.MapKeyToValue("Number", "Label"))

    // train the model
    let model = trData |> pipeline.Fit

    // get predictions and compare them to the ground truth
    let metrics = teData |> model.Transform |> mlcontext.MulticlassClassification.Evaluate

    // show metrics
    printfn "Параметры:"
    printfn "  MicroAccuracy:    %f" metrics.MicroAccuracy
    printfn "  MacroAccuracy:    %f" metrics.MacroAccuracy
    printfn "  LogLoss:          %f" metrics.LogLoss
    printfn "  LogLossReduction: %f" metrics.LogLossReduction

    // test data
    let digits = mlcontext.Data.CreateEnumerable(teData, reuseRowObject = false) |> Array.ofSeq
    let testDigits = [ digits.[9]; digits.[19]; digits.[8]; digits.[57]; digits.[109] ]

    let engine = mlcontext.Model.CreatePredictionEngine model

    // show predictions
    printfn "Предсказание модели:\n"
    printf "  Цифра\t\t"; [0..9] |> Seq.iter(fun i -> printf "%i\t\t" i)
    printfn ""
    testDigits |> Seq.iter(
        fun digit -> 
            printf "  %i\t\t" (int digit.Number)
            let p = engine.Predict digit
            p.Score |> Seq.iter (fun s -> printf "%f\t" s)
            printfn "")

    0 // return value