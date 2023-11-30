$(document).ready(function(){
     var val = document.getElementsByTagName("input")[0];
     switch(val.value){
     case 'p': 
         console.log("poison");
         $("#result").addClass("poisonous");
         break;
     case 'e' : console.log("e"); 
         $("#result").addClass("edible"); break;
     default: console.log("wrong"); break;
     }
     
});