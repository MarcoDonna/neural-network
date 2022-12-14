class ReLU{
    static f(x){
        return x > 0 ? x : 0;
    }

    static d(x){
        return x > 0 ? 1 : 0;
    }
}

class Sigmoid{
    static f(x){
        return 1 / (1 + Math.exp(-x));
    }

    static d(x){
        return this.f(x) * (1 - this.f(x));
    }
}

class Linear{
    static f(x){return x}
    static d(x){return 1}
}

