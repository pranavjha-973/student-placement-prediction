import joblib


model = joblib.load("model.pkl")
scale = joblib.load("scaler.pkl")


cgpa = float(input("Enter CGPA: "))
iq = float(input("Enter IQ: "))

data = [[cgpa, iq]]


data = scale.transform(data)

prediction = model.predict(data)

if prediction[0] == 1:
    print("Student will be placed")
else:
    print("Student will not be placed")