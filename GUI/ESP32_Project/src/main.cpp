#include <Arduino.h>


const int buttonPins[] = {D1, D2, D3, D5};
int buttonStates[4] = {0, 0, 0, 0};

void setup() {
    Serial.begin(115200);
    
    for (int i = 0; i < 4; i++) {
        pinMode(buttonPins[i], INPUT_PULLUP);
    }
}

void loop() {
    for (int i = 0; i < 4; i++) {
        buttonStates[i] = digitalRead(buttonPins[i]);

        if (buttonStates[i] == LOW) { 
            Serial.print("Button ");
            Serial.print(i + 1);
            Serial.println(" Pressed!");
            
            switch (i) {
                case 0:
                    Serial.println("upload");
                    break;
                case 1:
                    Serial.println("delete");
                    break;
                case 2:
                    Serial.println("rescan");
                    break;
                case 3:
                    Serial.println("confirm");
                    break;
            }
            delay(1000);
        }
    }
}
