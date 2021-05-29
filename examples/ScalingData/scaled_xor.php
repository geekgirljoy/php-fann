<?php
// This example will use the XOR dataset with negative one represented 
// as zero and one represented as one-hundred and demonstrate how to
// scale those values so that FANN can understand them and then how 
// to de-scale the value FANN returns so that you can understand them.

// Scaling allows you to take raw data numbers like -1234.975 or 4502012 
// in your dataset and convert them into an input/output range that
// your neural network can understand. 

// De-scaling lets you take the scaled data and convert it back into 
// the original range.


// scale_test.data
// Note the values are "raw" or un-scaled.
/*
4 2 1
0 0
0
0 100
100
100 0
100
100 100
0
*/


////////////////////
// Configure ANN  //
////////////////////

// New ANN
$ann = fann_create_standard_array(3, [2,3,1]);

// Set activation functions
fann_set_activation_function_hidden($ann, FANN_SIGMOID_SYMMETRIC);
fann_set_activation_function_output($ann, FANN_SIGMOID_SYMMETRIC);

// Read raw (un-scaled) training data from file
$train_data = fann_read_train_from_file("scale_test.data");

// Scale the data range to -1 to 1
fann_set_input_scaling_params($ann , $train_data, -1, 1);
fann_set_output_scaling_params($ann , $train_data, -1, 1);


///////////
// Train //
///////////

// Presumably you would train here (uncomment to perform training)...

// fann_train_on_data($ann, $train_data, 100, 10, 0.01);

// But it's not needed to test the scaling because the training file 
// in this case is just used to compute/derive the scale range. 
// However, doing the training will improve the answer the ANN gives
// in correlation to the training data.


//////////
// Test //
//////////

$raw_input = array(0, 100); // test XOR (0,100) input
$scaled_input = fann_scale_input ($ann , $raw_input); // scaled XOR (-1,1) input
$descaled_input = fann_descale_input ($ann , $scaled_input); // de-scaled XOR (0,100) input
$raw_output = fann_run($ann, $scaled_input); // get the answer/output from the ANN
$output_descale = fann_descale_output($ann, $raw_output); // de-scale the output 


////////////////////
// Report Results //
////////////////////
echo 'The raw_input:' . PHP_EOL;
var_dump($raw_input); 

echo 'The raw_input Scaled then De-Scaled (values are unchanged/correct):' . PHP_EOL;
var_dump($descaled_input); 

echo 'The Scaled input:' . PHP_EOL;
var_dump($scaled_input); 

echo "The raw_output of the ANN (Scaled input):" . PHP_EOL;
var_dump($raw_output);
 
echo 'The De-Scaled output:' . PHP_EOL;
var_dump($output_descale); 
 
 
////////////////////
// Example Output //
////////////////////

 /*
The raw_input:
array(2) {
  [0]=>
  float(0)
  [1]=>
  float(100)
}
The raw_input Scaled then De-Scaled (values are unchanged/correct):
array(2) {
  [0]=>
  float(0)
  [1]=>
  float(100)
}
The Scaled input:
array(2) {
  [0]=>
  float(-1)
  [1]=>
  float(1)
}
The raw_output of the ANN (Scaled input):
array(1) {
  [0]=>
  float(1)
}
The De-Scaled output:
array(1) {
  [0]=>
  float(100)
}
*/
