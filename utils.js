function dot(a, b){
    if(a.length != b.length)
        throw new Error('Invalid shape');

    let ret = 0;
    for(let i = 0; i < a.length; i++)
        ret += a[i] * b[i];
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