from ultralytics import YOLO
import cv2
import sys

# Define class names
class_names = [
    "Pomme", "Banane", "Livre", "Poivron", "Carotte", "Boite de cereales", "Ail",
    "Piment vert", "Kochujang", "Gombo", "Oignon", "Orange", "Pomme de terre",
    "Piment rouge", "Courge eponge", "Tomate", "pomme", "avocat", "bacon", "banane",
    "boeuf", "pain", "bardane", "beurre", "chou", "mais en conserve", "thon en conserve",
    "carotte", "fromage", "poulet", "poudre de piment", "pain au chocolat", "cannelle",
    "huile de cuisson", "mais", "cornflake", "chair de crabe", "concombre",
    "poudre de curry", "ravioli", "oeuf", "gateau de poisson", "frites", "ail",
    "gingembre", "oignon vert", "jambon", "galette de pommes de terre", "saucisse",
    "glace", "ketchup", "kimchi", "citron", "jus de citron", "mandarine", "guimauve",
    "mayonnaise", "lait", "mozzarella", "champignon", "moutarde", "chips nacho",
    "nouille", "nutella", "huile olive", "oignon", "oreo", "fromage parmesan",
    "persil", "pates", "beurre de cacahuete", "poire", "poivron", "poudre de poivre",
    "cornichon", "radis marine", "piment", "ananas", "porc", "pomme de terre", "ramen",
    "vin rouge", "riz", "sel", "saucisse", "algue", "sesame", "huile de sesame",
    "pate de crevettes", "sauce soja", "spam", "calamar", "fraise", "sucre",
    "patate douce", "tofu", "tomate", "wasabi", "pasteque", "creme fouettee",
]

# Load the YOLO model
model = YOLO('best.pt')

# Function to perform recognition and display results
def recognize_food(image_path):
    img = cv2.imread(image_path)
    
    # Perform inference
    results = model.predict(img)
    
    # Parse results and match with class names
    for result in results:
        for box in result.boxes:
            # Extract coordinates and class ID
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            confidence = box.conf[0]
            
            # Get the class name from the class ID
            label = class_names[class_id] if class_id < len(class_names) else "Unknown"
            
            # Draw bounding box and label on the image
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f'{label} {confidence:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Display the result
    cv2.imshow('Food Recognition', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Main execution block
if __name__ == "__main__":
    # Ensure an image path is provided
    if len(sys.argv) < 2:
        print("Usage: python recognize_food.py <path_to_image>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    recognize_food(image_path)

