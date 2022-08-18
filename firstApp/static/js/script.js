const inputFile = document.getElementById("inputFile")
const previewContainer = document.getElementById("imagePreview")
const previewImage = document.querySelector(".image-preview__image")
const previewDefaultText = document.querySelector(".image-preview__default-text")
const buttonHide = document.getElementById("btn-hide")

inputFile.addEventListener("change",function(){
    const file = this.files[0]
    console.log(file);

if(file){
    const reader = new FileReader();
    previewDefaultText.style.display = "none"
    previewImage.style.display = "block";
    previewImage.style.border = "none";
    buttonHide.style.display = "block";

    reader.addEventListener("load",function(){
        previewImage.setAttribute("src",this.result)
    })
    reader.readAsDataURL(file);
}else{
    previewDefaultText.style.display = null;
    previewImage.style.display = null;
}
})