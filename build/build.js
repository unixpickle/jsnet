(function() {
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
function Variable(value) {
    this.value = value;
    this.gradient = new Tensor(this.value.shape);
}

Variable.prototype.backward = function(outGrad) {
    this.gradient.add(outGrad);
};

Variable.prototype.clearGrad = function() {
    this.gradient = new Tensor(this.gradient.shape);
};

function pool(input, f) {
    var poolVar = new Variable(input.value);
    var result = f(poolVar);
    return {
        value: result.value,
        backward: function(outGrad) {
            poolVar.clearGrad();
            result.backward(outGrad);
            input.backward(poolVar.gradient);
        }
    };
}

function scale(input, scale) {
    return {
        value: input.value.copy().scale(scale),
        backward: function(outGrad) {
            input.backward(outGrad.copy().scale(scale));
        }
    };
}

function add(input1, input2) {
    return {
        value: input1.value.copy().add(input2.value),
        backward: function(outGrad) {
            input1.backward(outGrad);
            input2.backward(outGrad);
        }
    };
}

function sub(input1, input2) {
    return {
        value: input1.value.copy().sub(input2.value),
        backward: function(outGrad) {
            input1.backward(outGrad);
            input2.backward(outGrad.copy().scale(-1));
        }
    };
}

function mul(input1, input2) {
    return {
        value: input1.value.copy().mul(input2.value),
        backward: function(outGrad) {
            input1.backward(outGrad.copy().mul(input2.value));
            input2.backward(outGrad.copy().mul(input1.value));
        }
    };
}

function div(input1, input2) {
    return {
        value: input1.value.copy().div(input2.value),
        backward: function(outGrad) {
            input1.backward(outGrad.copy().div(input2.value));
            var denomGrad = outGrad.copy().mul(input1.value);
            denomGrad.div(input2.value).div(input2.value).scale(-1);
            input2.backward(denomGrad);
        }
    };
}

function reshape(input, shape) {
    return {
        value: input.value.copy().reshape(shape),
        backward: function(outGrad) {
            input.backward(outGrad.copy().reshape(input.value.shape));
        }
    };
}

function repeated(input, repeats) {
    return {
        value: input.value.repeated(repeats),
        backward: function(outGrad) {
            input.backward(outGrad.sumOuter());
        }
    };
}

function sumOuter(input) {
    if (input.value.shape.length === 0) {
        return input;
    }
    return {
        value: input.value.sumOuter(),
        backward: function(outGrad) {
            input.backward(outGrad.repeated(input.value.shape[0]));
        }
    };
}
var exportObj = {
    Tensor: Tensor,
    Variable: Variable,
    pool: pool,
    scale: scale,
    add: add,
    sub: sub,
    mul: mul,
    div: div,
    reshape: reshape,
    repeated: repeated,
    sumOuter: sumOuter
};

if ('undefined' !== typeof window) {
    window.jsnet = exportObj;
} else {
    module.exports = exportObj;
}
})();
