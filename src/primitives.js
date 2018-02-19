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
