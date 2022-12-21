class Car{
    constructor(par){
        this.x = par.x || 0;
        this.y = par.y || 0;
        this.on = true;
        this.width = par.width || 20;
        this.height = par.height || 30;
        this.angle = par.angle || 0;
        this.color = par.color || [255, 0, 0];
        this.immortal = false;
        this.alive = true;
        this.turn = 0.02;
        this.speed = 2;

        this.autopilot = {on: false, model: new NeuralNetwork([6, 4, 4, 2])}

        this.raycast = [];
        this.perimeter = [];
    }

    setPerimeter(p){
        this.perimeter = [[this.x - this.height/2, this.y - this.width/2, this.x + this.height/2, this.y - this.width/2],
                        [this.x - this.height/2, this.y - this.width/2, this.x - this.height/2, this.y + this.width/2],
                        [this.x + this.height/2, this.y + this.width/2, this.x - this.height/2, this.y + this.width/2],
                        [this.x + this.height/2, this.y + this.width/2, this.x + this.height/2, this.y - this.width/2]]
        return this;
    }

    move(m){
        m = m || 1;
        this.x -= (this.on ? 1 : 0) * Math.sin(this.angle) * this.speed * m;
        this.y += (this.on ? 1 : 0) * Math.cos(this.angle) * this.speed * m;
        return this;
    }

    control(l, r){
        this.angle += (this.on == true ? 1 : 0) * l * this.turn;
        this.angle -= (this.on == true ? 1 : 0) * r * this.turn;
        return this;
    }

    distance(env, cars){
        if(!cars)
            cars = [];
        const rays = [
            [this.x - Math.sin(this.angle-0.5) * 150, this.y + Math.cos(this.angle-0.5) * 150],
            [this.x - Math.sin(this.angle-0.3) * 150, this.y + Math.cos(this.angle-0.3) * 150],
            [this.x - Math.sin(this.angle-0.1) * 150, this.y + Math.cos(this.angle-0.1) * 150],
            //[this.x - Math.sin(this.angle) * 150, this.y + Math.cos(this.angle) * 150],                    
            [this.x - Math.sin(this.angle+0.1) * 150, this.y + Math.cos(this.angle+0.1) * 150],
            [this.x - Math.sin(this.angle+0.3) * 150, this.y + Math.cos(this.angle+0.3) * 150],
            [this.x - Math.sin(this.angle+0.5) * 150, this.y + Math.cos(this.angle+0.5) * 150],
        ]
        let rayData = [];
        for(let i = 0; i < rays.length; i++){
            let rayMapInterceptPoints = [];
            for(let j = 0; j < env.map.length; j++)                   
                rayMapInterceptPoints.push(intercept([this.x, this.y], rays[i], [env.map[j][0], env.map[j][1]], [env.map[j][2], env.map[j][3]]))
            
            for(let j = 0; j < cars.length; j++)
                for(let k = 0; k < cars[j].perimeter.length; k++)
                    rayMapInterceptPoints.push(intercept([this.x, this.y], rays[i], [cars[j].perimeter[k][0], cars[j].perimeter[k][1]], [cars[j].perimeter[k][2], cars[j].perimeter[k][3]]))
            rayData.push(closest([this.x, this.y], rayMapInterceptPoints.map(point => point === Infinity ? rays[i] : point)));
        }
        this.raycast = rayData;
        return this;
    }

    aliveStatus(){
        for(let i = 0; i < this.raycast.length; i++)
            if(euclideanDistance([this.x, this.y], this.raycast[i]) < Math.max(this.height/2, this.width/2) && this.immortal == false){
                this.alive = false;
                if(this.onDeath)
                    this.onDeath(this);
            }
        return this;
    }

    render(p, showRaycast){
        //Render Car
        p.translate(this.x, this.y);
        p.rotate(this.angle);
        p.rect(-this.width / 2, -this.height / 2, this.width, this.height);
        p.rect(-1, -1, 1, 1);                
        p.rotate(-this.angle);
        p.translate(-this.x, -this.y);  
        if(showRaycast)              
            for(let i = 0; i < this.raycast.length; i++){
                p.line(this.x, this.y, ...this.raycast[i]);
                p.ellipse(this.raycast[i][0], this.raycast[i][1], 4, 4);                    
            }
        return this;
    }
}