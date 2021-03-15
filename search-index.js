var searchIndex = JSON.parse('{\
"main":{"doc":"An example usage of the matrix library.","i":[[5,"main","main","The main function.",null,[[]]]],"p":[]},\
"reural_network":{"doc":"A simple neural network implementation.","i":[[0,"matrix","reural_network","A simple and naive implementation of mathematical …",null,null],[3,"Matrix","reural_network::matrix","A matrix is a 2-dimensional structure with specific …",null,null],[4,"Error","reural_network","A wrapper type for all errors caused by this crate.",null,null],[13,"CellOutOfBounds","","If an element is accessed whose coordinates (row and …",0,null],[13,"DimensionMismatch","","If the dimensions of a matrix do not match the dimensions …",0,null],[13,"DimensionsTooLarge","","If the dimensions of a matrix exceed the maximum allowed …",0,null],[13,"EmptyNetwork","","If a neural network is created without any layers, this …",0,null],[6,"Result","","A specialized <code>Result</code> type for Reural Network.",null,null],[3,"NeuralNetwork","","A neural network.",null,null],[3,"NeuralNetworkBuilder","","A builder for creating neural networks.",null,null],[11,"from","","",0,[[]]],[11,"into","","",0,[[]]],[11,"to_string","","",0,[[],["string",3]]],[11,"borrow","","",0,[[]]],[11,"borrow_mut","","",0,[[]]],[11,"try_from","","",0,[[],["result",4]]],[11,"try_into","","",0,[[],["result",4]]],[11,"type_id","","",0,[[],["typeid",3]]],[11,"vzip","","",0,[[]]],[11,"from","reural_network::matrix","",1,[[]]],[11,"into","","",1,[[]]],[11,"to_owned","","",1,[[]]],[11,"clone_into","","",1,[[]]],[11,"to_string","","",1,[[],["string",3]]],[11,"borrow","","",1,[[]]],[11,"borrow_mut","","",1,[[]]],[11,"try_from","","",1,[[],["result",4]]],[11,"try_into","","",1,[[],["result",4]]],[11,"type_id","","",1,[[],["typeid",3]]],[11,"vzip","","",1,[[]]],[11,"from","reural_network","",2,[[]]],[11,"into","","",2,[[]]],[11,"borrow","","",2,[[]]],[11,"borrow_mut","","",2,[[]]],[11,"try_from","","",2,[[],["result",4]]],[11,"try_into","","",2,[[],["result",4]]],[11,"type_id","","",2,[[],["typeid",3]]],[11,"vzip","","",2,[[]]],[11,"from","","",3,[[]]],[11,"into","","",3,[[]]],[11,"borrow","","",3,[[]]],[11,"borrow_mut","","",3,[[]]],[11,"try_from","","",3,[[],["result",4]]],[11,"try_into","","",3,[[],["result",4]]],[11,"type_id","","",3,[[],["typeid",3]]],[11,"vzip","","",3,[[]]],[11,"clone","reural_network::matrix","Clone this matrix.",1,[[]]],[11,"eq","","Check if two matrices are equal to each other.",1,[[]]],[11,"fmt","reural_network","",0,[[["formatter",3]],["result",6]]],[11,"fmt","reural_network::matrix","",1,[[["formatter",3]],["result",6]]],[11,"fmt","reural_network","",2,[[["formatter",3]],["result",6]]],[11,"fmt","","",3,[[["formatter",3]],["result",6]]],[11,"fmt","","Format this error using the given formatter.",0,[[["formatter",3]],["fmtresult",6]]],[11,"fmt","reural_network::matrix","Get a human readable representation of this matrix.",1,[[["formatter",3]],["result",6]]],[11,"div","","Divide each element in <code>self</code> by the corresponding element …",1,[[["matrix",3]]]],[11,"div","","Divide each element in <code>self</code> by the corresponding element …",1,[[["matrix",3]]]],[11,"div","","Divide each element in <code>self</code> by <code>other</code>.",1,[[]]],[11,"rem","","Calculate the remainder when dividing each element in <code>self</code>…",1,[[["matrix",3]]]],[11,"rem","","Calculate the remainder when dividing each element in <code>self</code>…",1,[[["matrix",3]]]],[11,"rem","","Calculate the remainder when dividing each element in <code>self</code>…",1,[[]]],[11,"sub","","Subtract each element in <code>other</code> from the corresponding …",1,[[["matrix",3]]]],[11,"sub","","Subtract each element in <code>other</code> from the corresponding …",1,[[["matrix",3]]]],[11,"sub","","Subtract <code>other</code> from all elements in <code>self</code>.",1,[[]]],[11,"add","","Add each element in <code>self</code> to the corresponding element in …",1,[[["matrix",3]]]],[11,"add","","Add each element in <code>self</code> to the corresponding element in …",1,[[["matrix",3]]]],[11,"add","","Add <code>other</code> to all elements in <code>self</code>.",1,[[]]],[11,"mul","","Multiply each element in <code>self</code> to the corresponding …",1,[[["matrix",3]]]],[11,"mul","","Multiply each element in <code>self</code> to the corresponding …",1,[[["matrix",3]]]],[11,"mul","","Multiply each element in <code>self</code> by <code>other</code>.",1,[[]]],[11,"neg","","Negate all elements in <code>self</code>.",1,[[]]],[11,"add_assign","","Add <code>other</code> to all elements in <code>self</code>.",1,[[]]],[11,"sub_assign","","Subtract <code>other</code> from all elements in <code>self</code>.",1,[[]]],[11,"mul_assign","","Multiply each element in <code>self</code> by <code>other</code>.",1,[[]]],[11,"div_assign","","Divide each element in <code>self</code> by <code>other</code>.",1,[[]]],[11,"rem_assign","","Calculate the remainder when dividing each element in <code>self</code>…",1,[[]]],[11,"not","","Logically negate all elements in <code>self</code>.",1,[[]]],[11,"bitand","","Calculate the bitwise AND of each element in <code>self</code> with …",1,[[["matrix",3]]]],[11,"bitand","","Calculate the bitwise AND of each element in <code>self</code> with …",1,[[["matrix",3]]]],[11,"bitand","","Calculate the bitwise AND of each element in <code>self</code> with …",1,[[]]],[11,"bitor","","Calculate the bitwise OR of each element in <code>self</code> with the …",1,[[["matrix",3]]]],[11,"bitor","","Calculate the bitwise OR of each element in <code>self</code> with the …",1,[[["matrix",3]]]],[11,"bitor","","Calculate the bitwise OR of each element in <code>self</code> with …",1,[[]]],[11,"bitxor","","Calculate the bitwise XOR of each element in <code>self</code> with …",1,[[["matrix",3]]]],[11,"bitxor","","Calculate the bitwise XOR of each element in <code>self</code> with …",1,[[["matrix",3]]]],[11,"bitxor","","Calculate the bitwise XOR of each element in <code>self</code> with …",1,[[]]],[11,"shl","","Bitwise shift each element in <code>self</code> to the left by the …",1,[[["matrix",3]]]],[11,"shl","","Bitwise shift each element in <code>self</code> to the left by the …",1,[[["matrix",3]]]],[11,"shl","","Bitwise shift each element in <code>self</code> to the left by <code>other</code>.",1,[[]]],[11,"shr","","Bitwise shift each element in <code>self</code> to the right by the …",1,[[["matrix",3]]]],[11,"shr","","Bitwise shift each element in <code>self</code> to the right by the …",1,[[["matrix",3]]]],[11,"shr","","Bitwise shift each element in <code>self</code> to the right by <code>other</code>.",1,[[]]],[11,"bitand_assign","","Calculate the bitwise AND of each element in <code>self</code> with …",1,[[]]],[11,"bitor_assign","","Calculate the bitwise OR of each element in <code>self</code> with …",1,[[]]],[11,"bitxor_assign","","Calculate the bitwise XOR of each element in <code>self</code> with …",1,[[]]],[11,"shl_assign","","Bitwise shift each element in <code>self</code> to the left by <code>other</code>.",1,[[]]],[11,"shr_assign","","Bitwise shift each element in <code>self</code> to the right by <code>other</code>.",1,[[]]],[11,"source","reural_network","The underlying source of this error, if any.",0,[[],[["option",4],["error",8]]]],[11,"as_slice","reural_network::matrix","Get the data of the matrix as a 1-dimensional slice.",1,[[]]],[11,"get_number_of_columns","","Get the number of columns in the matrix.",1,[[]]],[11,"get_number_of_rows","","Get the number of rows in the matrix.",1,[[]]],[11,"map_ref_mut","","Mutate each element in the matrix in place as given by …",1,[[]]],[11,"new","","Create a new matrix with the given dimensions and the …",1,[[["nonzerousize",3]],[["result",6],["matrix",3]]]],[11,"from_slice","","Convert a slice into a matrix of the given dimensions.",1,[[["nonzerousize",3]],[["result",6],["matrix",3]]]],[11,"get","","Get the value in the given <code>row</code> and <code>column</code>.",1,[[],["result",6]]],[11,"get_unchecked","","Get the value in the given <code>row</code> and <code>column</code>.",1,[[]]],[11,"map","","Map each element in the matrix to a new element as given …",1,[[]]],[11,"transpose","","Transpose this matrix.",1,[[],["matrix",3]]],[11,"matrix_mul","","Compute the matrix product of <code>self</code> and <code>other</code> and return …",1,[[["matrix",3]],[["result",6],["matrix",3]]]],[11,"from_random","","Create a new matrix with the given dimensions and random …",1,[[["nonzerousize",3]],[["result",6],["matrix",3]]]],[11,"predict","reural_network","Let the neural network predict an output for the given …",2,[[["matrix",3]],[["result",6],["matrix",3]]]],[11,"new","","Start building a new neural network with the given number …",3,[[["nonzerousize",3]]]],[11,"add_hidden_layer","","Add a hidden layer with the given number of <code>nodes</code> to the …",3,[[["nonzerousize",3]]]],[11,"add_output_layer","","Add an output layer with the given number of nodes to the …",3,[[["nonzerousize",3]],[["result",6],["neuralnetwork",3]]]]],"p":[[4,"Error"],[3,"Matrix"],[3,"NeuralNetwork"],[3,"NeuralNetworkBuilder"]]}\
}');
addSearchOptions(searchIndex);initSearch(searchIndex);