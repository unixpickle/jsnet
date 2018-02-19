function padImages(images, top, right, bottom, left) {
    if (images.value.shape.length !== 4) {
        throw Error('expected 4-D image tensor');
    }
    var padded = new Tensor([
        images.value.shape[0],
        images.value.shape[1] + top + bottom,
        images.value.shape[2] + left + right,
        images.value.shape[3]
    ]);
    for (var i = 0; i < images.value.shape[0]; ++i) {
        var srcIdx = i * shapeProduct(images.value.shape.slice(1));
        var dstIdx = i * shapeProduct(padded.shape.slice(1));
        for (var j = 0; j < images.value.shape[1]; ++j) {
            var jRowSize = shapeProduct(images.value.shape.slice(2));
            var srcStart = srcIdx + j * jRowSize;
            var dstStart = (dstIdx + (j + top) * shapeProduct(padded.shape.slice(2)) +
                left * padded.shape[3]);
            for (var k = 0; k < jRowSize; ++k) {
                padded.data[dstStart + k] = images.value.data[srcStart + k];
            }
        }
    }
    return {
        value: padded,
        backward: function(outGrad) {
            var unpadded = new Tensor(images.value.shape);
            for (var i = 0; i < unpadded.shape[0]; ++i) {
                var srcIdx = i * shapeProduct(outGrad.shape.slice(1));
                var dstIdx = i * shapeProduct(unpadded.shape.slice(1));
                for (var j = 0; j < unpadded.shape[1]; ++j) {
                    var jRowSize = shapeProduct(unpadded.shape.slice(2));
                    var dstStart = dstIdx + j * jRowSize;
                    var srcStart = (srcIdx + (j + top) * shapeProduct(outGrad.shape.slice(2)) +
                        left * outGrad.shape[3]);
                    for (var k = 0; k < jRowSize; ++k) {
                        unpadded.data[dstStart + k] = outGrad.data[srcStart + k];
                    }
                }
            }
            images.backward(unpadded);
        }
    };
}

function imagePatches(images, windowHeight, windowWidth, strideY, strideX) {
    if (images.value.shape.length !== 4) {
        throw Error('expected 4-D image tensor');
    } else if (images.value.shape[1] < windowHeight || images.value.shape[2] < windowWidth) {
        throw Error('window larger than image');
    }
    var rowSize = shapeProduct(images.value.shape.slice(2));
    var depth = images.value.shape[3];
    var result = new Tensor([
        images.value.shape[0],
        1 + Math.floor((images.value.shape[1] - windowHeight) / strideY),
        1 + Math.floor((images.value.shape[2] - windowWidth) / strideX),
        windowHeight,
        windowWidth,
        depth
    ]);
    var dstIdx = 0;
    for (var i = 0; i < result.shape[0]; ++i) {
        var batchStart = i * shapeProduct(images.value.shape.slice(1));
        for (var j = 0; j < result.shape[1]; ++j) {
            for (var k = 0; k < result.shape[2]; ++k) {
                var srcIdx = batchStart + (j * strideY * rowSize) + (k * strideX * depth);
                for (var l = 0; l < windowHeight; ++l) {
                    var rowStart = srcIdx + l * rowSize;
                    for (var m = 0; m < windowWidth * depth; ++m) {
                        result.data[dstIdx++] = images.value.data[rowStart + m];
                    }
                }
            }
        }
    }
    return {
        value: result,
        backward: function(outGrad) {
            var inGrad = new Tensor(images.value.shape);
            var outIdx = 0;
            for (var i = 0; i < result.shape[0]; ++i) {
                var batchStart = i * shapeProduct(images.value.shape.slice(1));
                for (var j = 0; j < result.shape[1]; ++j) {
                    for (var k = 0; k < result.shape[2]; ++k) {
                        var srcIdx = batchStart + (j * strideY * rowSize) + (k * strideX * depth);
                        for (var l = 0; l < windowHeight; ++l) {
                            var rowStart = srcIdx + l * rowSize;
                            for (var m = 0; m < windowWidth * depth; ++m) {
                                inGrad.data[rowStart + m] += outGrad.data[outIdx++];
                            }
                        }
                    }
                }
            }
            images.backward(inGrad);
        }
    };
}
