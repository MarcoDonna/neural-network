function dot(a, b){
    if(a.length != b.length)
        throw new Error('Invalid shape');

    let ret = 0;
    for(let i = 0; i < a.length; i++)
        ret += (a[i] === undefined ? 0 : a[i]) * (b[i] === undefined ? 0 : b[i]);
    return ret;
}

function add(vector, value=0){
    if(typeof value == 'object')
        if(vector.length == value.length)
            return vector.map((item, index) => item+value[index])
        else
            throw new Error('Invalid shape');
    return vector.map(item => item+value);
}

function sum(vector){
    return vector.reduce((acc, val) => acc + val);
}

function avg(vector){
    return sum(vector) / vector.length;
}

function flatten(matrix){
    let ret = [];
    for(let i = 0; i < matrix.length; i++)
        ret.push(...matrix[i]);
    return ret;
}

if(typeof module !== 'undefined' && module.exports)
    module.exports = {dot, add, sum, avg, flatten};