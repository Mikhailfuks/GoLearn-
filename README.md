package main

import ("fmt")

 "github.com/sjwhitworth/golearn/base"
 "github.com/sjwhitworth/golearn/evaluation"
 "github.com/sjwhitworth/golearn/linear_models"
 "github.com/sjwhitworth/golearn/preprocess"
)

func main() {
 // Create a dataset
 dataset := base.NewDenseInstances([][]float64{
  {1, 2, 3},
  {4, 5, 6},
  {7, 8, 9},
 }, [][]float64{
  {10},
  {20},
  {30},
 })

 // Split the dataset into training and testing sets
 trainData, testData := base.SplitInstances(dataset, 0.7) // 70% for training

 // Preprocess the data
 preprocess.RemoveConstantAttributes(trainData)
 preprocess.RemoveConstantAttributes(testData)
 preprocess.Normalize(trainData)
 preprocess.Normalize(testData)

 // Create a linear regression model
 model := linear_models.NewLinearRegression(linear_models.L2Regularization, 0.1) // L2 regularization with coefficient 0.1

 // Train the model
 model.Fit(trainData)

 // Evaluate the model on the test data
 predictions, err := model.Predict(testData)
 if err != nil {
  panic(err)
 }

 // Calculate the accuracy
 accuracy, err := evaluation.GetAccuracy(testData, predictions)
 if err != nil {
  panic(err)
 }

 // Print the results
 fmt.Println("Accuracy:", accuracy)

 // Get the model coefficients
 coefficients := model.GetCoefficients()
 fmt.Println("Coefficients:", coefficients)
}
