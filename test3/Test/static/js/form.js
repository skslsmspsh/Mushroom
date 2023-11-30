$(document).ready(function(){
    console.log("hi, I'm ready.");
});

$("#dv1").click(function(){		
        var isActive = $("#dv1").hasClass("active");
        var val = document.getElementById("dv1").getElementsByTagName("input")[0];
        if(isActive){
            $("#dv1").removeClass("active");
            $(val).prop("checked", false);
        }
        else{
            $("#la1").removeClass("active");
            $("#la2").removeClass("active");
            $("#la3").removeClass("active");
            $("#dv2").removeClass("active");
            $("#dv3").removeClass("active");
            $("#dv1").addClass("active");
            $(val).prop("checked", true);
        }
    }
)
$("#dv2").click(function(){		
        var isActive = $("#dv2").hasClass("active");
        var val = document.getElementById("dv2").getElementsByTagName("input")[0];
        
        if(isActive){
            $("#dv2").removeClass("active");
            $(val).prop("checked",false);
        }
        else{
            $("#la1").removeClass("active");
            $("#la2").removeClass("active");
            $("#la3").removeClass("active");
            $("#dv1").removeClass("active");
            $("#dv3").removeClass("active");
            $("#dv2").addClass("active");
            $(val).prop("checked",true);
        }
    }
)
$("#dv3").click(function(){		
        var isActive = $("#dv3").hasClass("active");
         var val = document.getElementById("dv3").getElementsByTagName("input")[0];
         
        if(isActive){
            $("#dv3").removeClass("active");
            $(val).prop("checked",false);
        }
        else{
            $("#la1").removeClass("active");
            $("#la2").removeClass("active");
            $("#la3").removeClass("active");
            $("#dv1").removeClass("active");
            $("#dv2").removeClass("active");
            $("#dv3").addClass("active");
            $(val).prop("checked",true);
        }
    }
)

$("#la1").click(function(){		
        var isActive = $("#la1").hasClass("active");
        var val = document.getElementById("la1").getElementsByTagName("input")[0];
        if(isActive){
            $("#la1").removeClass("active");
            $(val).prop("checked", false);
        }
        else{
            $("#dv1").removeClass("active");
            $("#dv2").removeClass("active");
            $("#dv3").removeClass("active");
            $("#la2").removeClass("active");
            $("#la3").removeClass("active");
            $("#la1").addClass("active");
            $(val).prop("checked", true);
        }
    }
)

$("#la2").click(function(){		
        var isActive = $("#la2").hasClass("active");
        var val = document.getElementById("la2").getElementsByTagName("input")[0];
        if(isActive){
            $("#la2").removeClass("active");
            $(val).prop("checked", false);
        }
        else{
            $("#dv1").removeClass("active");
            $("#dv2").removeClass("active");
            $("#dv3").removeClass("active");
            $("#la1").removeClass("active");
            $("#la3").removeClass("active");
            $("#la2").addClass("active");
            $(val).prop("checked", true);
        }
    }
)

$("#la3").click(function(){		
        var isActive = $("#la3").hasClass("active");
        var val = document.getElementById("la3").getElementsByTagName("input")[0];
        if(isActive){
            $("#la3").removeClass("active");
            $(val).prop("checked", false);
        }
        else{
            $("#dv1").removeClass("active");
            $("#dv2").removeClass("active");
            $("#dv3").removeClass("active");
            $("#la1").removeClass("active");
            $("#la2").removeClass("active");
            $("#la3").addClass("active");
            $(val).prop("checked", true);
        }
    }
)