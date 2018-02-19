function Tensor(shape, data) {
    var dataSize = shapeProduct(shape);
    this.shape = shape;
    if ('undefined' !== typeof Float32Array) {
        if (!data) {
            this.data = new Float32Array(dataSize);
        } else {
            if (data.length !== dataSize) {
                throw Error('invalid data size');
            }
            this.data = new Float32Array(data);
        }
    } else {
        if (!data) {
            this.data = [];
            for (var i = 0; i < dataSize; ++i) {
                this.data.push(0);
            }
        } else {
            if (data.length !== dataSize) {
                throw Error('invalid data size');
            }
            this.data = [];
            for (var i = 0; i < dataSize; ++i) {
                this.data.push(data[i]);
            }
        }
    }
}

Tensor.prototype.copy = function() {
    return new Tensor(this.shape, this.data);
};

Tensor.prototype.scale = function(scale) {
    for (var i = 0; i < this.data.length; ++i) {
        this.data[i] *= scale;
    }
    return this;
};

Tensor.prototype.add = function(other) {
    this._assertSameShape(other);
    for (var i = 0; i < this.data.length; ++i) {
        this.data[i] += other.data[i];
    }
    return this;
};

Tensor.prototype.sub = function(other) {
    this._assertSameShape(other);
    for (var i = 0; i < this.data.length; ++i) {
        this.data[i] -= other.data[i];
    }
    return this;
};

Tensor.prototype.mul = function(other) {
    this._assertSameShape(other);
    for (var i = 0; i < this.data.length; ++i) {
        this.data[i] *= other.data[i];
    }
    return this;
};

Tensor.prototype.div = function(other) {
    this._assertSameShape(other);
    for (var i = 0; i < this.data.length; ++i) {
        this.data[i] /= other.data[i];
    }
    return this;
};

Tensor.prototype.reshape = function(shape) {
    if (shapeProduct(shape) !== shapeProduct(this.shape)) {
        throw Error('cannot reshape from [' + this.shape + '] to [' + shape + ']');
    }
    this.shape = shape;
    return this;
};

Tensor.prototype.repeated = function(repeats) {
    var result = new Tensor([repeats].concat(this.shape));
    for (var i = 0; i < repeats; ++i) {
        var startIdx = i * this.data.length;
        for (var j = 0; j < this.data.length; ++j) {
            result.data[startIdx + j] = this.data[j];
        }
    }
    return result;
};

Tensor.prototype.sumOuter = function() {
    if (this.shape.length === 0) {
        return this.copy();
    }
    var result = new Tensor(this.shape.slice(1));
    var chunkSize = this.data.length / this.shape[0];
    for (var i = 0; i < this.shape[0]; ++i) {
        var offset = chunkSize * i;
        for (var j = 0; j < chunkSize; ++j) {
            result.data[j] += this.data[j + offset];
        }
    }
    return result;
};

Tensor.prototype._assertSameShape = function(other) {
    if (this.shape.length !== other.shape.length) {
        throw Error('shapes not equal');
    }
    for (var i = 0; i < this.shape.length; ++i) {
        if (this.shape[i] !== other.shape[i]) {
            throw Error('shapes not equal');
        }
    }
};

function shapeProduct(shape) {
    var dataSize = 1;
    for (var i = 0; i < shape.length; ++i) {
        dataSize *= shape[i];
    }
    return dataSize;
}
