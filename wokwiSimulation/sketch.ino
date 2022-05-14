#include <dht.h>
#include <LiquidCrystal_I2C.h>
#include <Servo.h>
#include <BasicLinearAlgebra.h>

using namespace BLA;

dht DHT[4];
LiquidCrystal_I2C lcd(0x27, 16, 2);
Servo servo[4];

void setup() {
  lcd.init();
  lcd.backlight();  
  lcd.setCursor(0, 0);
  lcd.print("1. ");
  lcd.setCursor(8, 0);
  lcd.print("2. ");
  lcd.setCursor(0, 1);
  lcd.print("3. ");
  lcd.setCursor(8, 1);
  lcd.print("4. ");

  for (int i = 0; i < 4; ++i) {
    servo[i].attach(servo_pin(i));
  }
}

void loop() {
  float humidity[4];
  float temp[4];
  float percentage[4];

  // read humidity and temperature
  for (int i = 0; i < 4; ++i) {
    DHT[i].read22(dht_pin(i));
    humidity[i] = DHT[i].humidity;
    temp[i] = DHT[i].temperature;
  }

  if (is_night()){
    for (int i = 0; i < 4; ++i) {
      percentage[i] = 0;
    }
  } else {
    // use multi-layer perceptron
    for (int i = 0; i < 4; ++i) {
      percentage[i] = predict_percentage(humidity[i], temp[i]);
    }
  }

  lcd.setCursor(3, 0);
  lcd.print(percentage[0], 0);
  lcd.print("% ");
  lcd.setCursor(11, 0);
  lcd.print(percentage[1], 0);
  lcd.print("% ");
  lcd.setCursor(3, 1);
  lcd.print(percentage[2], 0);
  lcd.print("% ");
  lcd.setCursor(11, 1);
  lcd.print(percentage[3], 0);
  lcd.print("% ");

  for (int i = 0; i < 4; ++i) {
    servo[i].write(percent_to_angle(percentage[i]));
  }

  delay(2000);
}

int percent_to_angle(float p) {
  return (int) 180 * (p / 100);
}

// pin number of ith DHT22 sensor
int dht_pin(int i) {
  return i+2;
}

// pin number of ith Servo motor
int servo_pin(int i) {
  return 13-i;
}

bool is_night() {
  // LDR Characteristics
  const float GAMMA = 0.7;
  const float RL10 = 50;

  int analogValue = analogRead(A1);
  float voltage = analogValue / 1024. * 5;
  float resistance = 2000 * voltage / (1 - voltage / 5);
  float lux = pow(RL10 * 1e3 * pow(10, GAMMA) / resistance, (1 / GAMMA));

  return lux < 50;
}

template <int R, int C>
BLA::Matrix<R, C> relu(BLA::Matrix<R, C> x) {
  for (int i = 0; i < R; ++i) {
    for (int j = 0; j < C; ++j) {
      if (x(i, j) < 0) {
        x(i, j) = 0.;
      }
    }
  }
  return x;
}

float predict_percentage(float humidity, float temp) {
  BLA::Matrix<2, 3> w0 = {-1.62512530e+00, -1.25647664e-01, 1.27342808e-03,
         1.30754050e+00, 2.22294306e-02, -3.04635327e-02};
  BLA::Matrix<3, 3> w1 = {-1.38935249e-063, 6.38250945e-001, -2.01915742e-100,
         6.21648834e-138, -3.83377990e+000, 4.18442800e-053,
         1.28126468e-047, -8.50167740e-002, -2.64964214e-041};
  BLA::Matrix<3, 1> w2 = {1.65620559e-44, 2.24336547e+00, -1.29960621e-36};

  BLA::Matrix<1, 3> b0 = {-3.03832916,  4.55312463,  0.21453741};
  BLA::Matrix<1, 3> b1 = {-0.67005969, -1.71418843, -0.53256778};
  BLA::Matrix<1, 1> b2 = {0.10515677};

  BLA::Matrix<1, 2> features = {humidity, temp};

  BLA::Matrix<1, 1> res = relu(relu(features*w0 + b0)*w1 + b1)*w2 + b2;

  float ans = res(0, 0);

  // if prediction is negative or > 100, project it to 0 and 100 respectively
  if (ans < 0) {
    ans = 0;
  } else if (ans > 100) {
    ans = 100;
  }
  return ans;
}