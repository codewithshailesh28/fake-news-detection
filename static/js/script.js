const textarea=document.getElementById("news");

const counter=document.getElementById("counter");

textarea.addEventListener("input",()=>{

counter.innerHTML=textarea.value.length+" Characters";

});

const form=document.querySelector("form");

form.addEventListener("submit",()=>{

document.getElementById("btnText").style.display="none";

document.getElementById("loading").style.display="inline";

});