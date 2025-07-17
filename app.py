import torch
from torchvision import transforms
from PIL import Image
from flask import Flask, render_template, request, redirect
import os
from datasets import classes
from model import model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

model.fc = torch.nn.Sequential(
    torch.nn.Linear(in_features=512, out_features=128),
    torch.nn.LeakyReLU(),
    torch.nn.Linear(in_features=128, out_features=10),
    torch.nn.Softmax(dim=1),
)
model.requires_grad_(False)
model = model.to('cpu')

model.layer4.requires_grad_(True)
model.layer3.requires_grad_(True)
model.fc.requires_grad_(True)

state_dict = torch.load('model2.pth', map_location='cpu')
model.load_state_dict(state_dict, strict=False)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.4861202538013458, 0.48226398229599, 0.4337790906429291],
                [0.2418711632490158, 0.23503021895885468, 0.25225475430488586])
])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def process_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img_t = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        output = model(img_t)
        _, predicted = torch.max(output, 1)
        return predicted.item()

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            print('No file selected')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            print('No file selected')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            

            result = process_image(filepath)
            try:
                # Обработка изображения
                img = Image.open(filepath).convert('RGB')
                img_t = transform(img).unsqueeze(0)
                
                with torch.no_grad():
                    output = model(img_t)
                    _, predicted = torch.max(output, 1)
                    class_id = predicted.item()
                    class_name = classes[class_id]
                
                # Показываем результат на этой же странице
                return render_template('index.html', 
                                    image_path=filepath,
                                    result=class_name,
                                    show_result=True)
            
            except Exception as e:
                print(f'Error processing image: {str(e)}')
                return f"Error: {str(e)}", -1
    
    # GET-запрос или ошибка - просто показываем форму
    return render_template('index.html', show_result=False)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)  
    app.run(debug=True)
