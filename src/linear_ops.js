function matmul(mat1, mat2) {
    if (mat1.value.shape.length !== 2 || mat2.value.shape.length !== 2) {
        throw Error('matrices must be two-dimensional');
    } else if (mat1.value.shape[1] !== mat2.value.shape[0]) {
        throw Error('inner dimension mismatch');
    }
    var resultData = new Tensor([mat1.value.shape[0], mat2.value.shape[1]]);
    var idx = 0;
    for (var i = 0, rows = mat1.value.shape[0]; i < rows; ++i) {
        var rowOffset = mat1.value.shape[1] * i;
        for (var j = 0, cols = mat2.value.shape[1]; j < cols; ++j) {
            var sum = 0;
            for (var k = 0, inner = mat1.value.shape[1]; k < inner; ++k) {
                sum += mat1.value.data[rowOffset + k] * mat2.value.data[k*cols + j];
            }
            resultData.data[idx++] = sum;
        }
    }
    return {
        value: resultData,
        backward: function(outGrad) {
            var mat1Grad = new Tensor(mat1.value.shape);
            var mat2Grad = new Tensor(mat2.value.shape);
            var idx = 0;
            for (var i = 0, rows = mat1.value.shape[0]; i < rows; ++i) {
                var rowOffset = mat1.value.shape[1] * i;
                for (var j = 0, cols = mat2.value.shape[1]; j < cols; ++j) {
                    var elemGrad = outGrad.data[idx++];
                    for (var k = 0, inner = mat1.value.shape[1]; k < inner; ++k) {
                        mat1Grad.data[rowOffset + k] += elemGrad * mat2.value.data[k*cols + j];
                        mat2Grad.data[k*cols + j] += elemGrad * mat1.value.data[rowOffset + k];
                    }
                }
            }
            mat1.backward(mat1Grad);
            mat2.backward(mat2Grad);
        }
    };
}

function conv2d(images, filters, strideY, strideX) {
    if (filters.value.shape.length !== 4) {
        throw new Error('expected 4-D filter tensor');
    }
    strideY = (strideY || 1);
    strideX = (strideX || strideY);
    var patches = imagePatches(images, filters.value.shape[0], filters.value.shape[1],
        strideY, strideX);
    var leftMatrix = reshape(patches, [shapeProduct(patches.value.shape.slice(0, 3)),
        shapeProduct(patches.value.shape.slice(3))]);
    var rightMatrix = reshape(filters, [shapeProduct(filters.value.shape.slice(0, 3)),
        filters.value.shape[3]]);
    var product = matmul(leftMatrix, rightMatrix);
    return reshape(product, patches.value.shape.slice(0, 3).concat([filters.value.shape[3]]));
}
