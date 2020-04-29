const spawn = require("child_process").spawn;

 function predict(fn, pdf){
     console.log(__dirname)
     const pythonProcess = spawn('python',[__dirname + "\\CAPS2\\3_detect.py", pdf]);
     pythonProcess.stdout.on('data', function(data){
        fn(data.toString())
    });
}




function call(){
    return predict(function(data){
       console.log(data) 
    }, __dirname + '\\grade.pdf')
}

call()

module.exports = predict