#include <PubSubClient.h>
#include <SPI.h>
#include <SD.h>
#include <WiFi.h>
#include <cmath>
#include <cstdlib>
#include <time.h>
#include <vector>
#include "KMeans.h"

#define PIN_ULTRASONIC_TRIG 4
#define PIN_ULTRASONIC_ECHO 16
#define PIN_BUZZER          0
#define PIN_FLOW_SENSOR     35

#define PIN_SD_CS           5
#define PIN_SD_SCK          18
#define PIN_SD_MISO         19
#define PIN_SD_MOSI         23
#define PIN_SD_CD           34 // Card Detect (Wired, but optional for basic SD usage)

#define ANOMALY_THRESHOLD   15.0f
#define LOG_BUFFER_SIZE     2048
#define LOOP_DURATION_MS    5000
#define MAX_TRAINING_LINES  2000

// WiFi
const char* ssid = "Wokwi-GUEST";
const char* password = "";

// MQTT
const char* mqttServer = "broker.hivemq.com"; // Refer to https://www.mqtt-dashboard.com/
const char* mqttResourcesTopic = "esp32/water/resources";
const char* mqttSensorLogTopic = "esp32/water/sensor_log";
const char* mqttSensorTopic = "esp32/water/sensors";
const char* mqttTrainingTopic = "esp32/water/training_window";

// NTP
const char* ntpServer = "pool.ntp.org";
const long gmtOffset_sec = 0; // Use 0 offset and let the dashboard handle it
const int daylightOffset_sec = 0; // Use 0 offset and let the dashboard handle it

// Directories
const char* aiToggleDir = "/ai_toggle.txt";
const char* sensorLogDir = "/sensor_log.csv";
const char* trainedModelDir = "/trained_model.bin";
const char* trainingWindowDir = "/training_window.txt";

WiFiClient espClient;
PubSubClient client(espClient);

KMeans kmeans(3, 100, 1e-4f);

void setup() {
  Serial.begin(115200);

  pinMode(PIN_ULTRASONIC_TRIG, OUTPUT);
  pinMode(PIN_ULTRASONIC_ECHO, INPUT);
  pinMode(PIN_BUZZER, OUTPUT);
  pinMode(PIN_FLOW_SENSOR, INPUT);
  pinMode(PIN_SD_CD, INPUT);

  SPI.begin(PIN_SD_SCK, PIN_SD_MISO, PIN_SD_MOSI, PIN_SD_CS);
  Serial.println("[SD] Mounting SD Card...");

  if (SD.begin(PIN_SD_CS, SPI)) {
    Serial.println("[SD] SD Card mounted successfully!");

    initFiles();

    if (getAiToggleValue()) {
      applyTraining();
    }
  } else {
    Serial.println("[ERROR] SD Card mount failed!");
  }

  Serial.print("[WIFI] Connecting to WiFi");
  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.print(".");
  }

  Serial.println("\n[WIFI] Connected to WiFi!");

  initTime();
  client.setServer(mqttServer, 1883);
  client.setBufferSize(LOG_BUFFER_SIZE);
  client.setCallback(mqttCallback);
}

void loop() {
  unsigned long startWait = millis();

  while (millis() - startWait < LOOP_DURATION_MS) {
    if (!client.connected()) {
      mqttReconnect();
    }

    client.loop();
    delay(50);
  }

  time_t epochTime = getTime();
  float distance = getDistanceToWater();
  int flowRaw = analogRead(PIN_FLOW_SENSOR);

  sendAlert(distance, flowRaw);
  logToSD(epochTime, distance, flowRaw);
  mqttPublishResources();
  mqttPublishTelemetry(epochTime, distance, flowRaw);
}

void mqttCallback(char* topic, byte* payload, unsigned int length) {
  String message = "";

  for (unsigned int i = 0; i < length; i++) {
    message += (char)payload[i];
  }

  message.trim();
  Serial.printf("[MQTT] Message arrived on [%s]: %s\n", topic, message.c_str());

  if (String(topic) == mqttTrainingTopic) {
    if (SD.exists(trainingWindowDir)) {
      SD.remove(trainingWindowDir);
    }

    File file = SD.open(trainingWindowDir, FILE_WRITE);

    if (file) {
      file.println(message);
      file.close();
      Serial.println("[SD] Training window updated successfully from MQTT!");

      if (getAiToggleValue()) {
        Serial.println("[KMEANS] Retraining with the new training window...");
        applyTraining();
      }
    } else {
      Serial.println("[ERROR] Failed to write training window to SD card.");
    }
  } else if (String(topic) == mqttSensorLogTopic) {
    int cursor = message.toInt();
    if (cursor <= 0) cursor = 1;

    const int LINES_PER_PAGE = 50;
    int startIndex = (cursor - 1) * LINES_PER_PAGE;

    File logFile = SD.open(sensorLogDir, FILE_READ);

    if (!logFile) {
      Serial.println("[ERROR] Failed to open sensor log file.");
      client.publish(mqttSensorLogTopic, "[ERROR] Log file not found or empty.");
      return;
    }

    int currentLine = 0;
    int sentCount = 0;
    bool hasMore = false;
    String payload = "";

    String startWrapper = "[LOG_START: cursor=" + String(cursor) + "]\n";
    String endWrapperBase = "\n[LOG_DUMP_END: cursor=" + String(cursor) + ", has_more=";
    int metadataOverhead = startWrapper.length() + endWrapperBase.length() + 8; // +8 for boolean flag room

    while (logFile.available()) {
      String line = logFile.readStringUntil('\n');
      line.trim();

      if (line.length() > 0) {
        if (currentLine >= startIndex) {
          if (startWrapper.length() + payload.length() + line.length() + 1 + metadataOverhead > LOG_BUFFER_SIZE) {
            hasMore = true;
            break;
          }

          payload += line + "\n";
          sentCount++;
        }

        currentLine++;
      }
    }

    logFile.close();

    String message = startWrapper +
                     payload +
                     "[LOG_END: cursor=" + String(cursor) + ", has_more=" + (hasMore ? "true" : "false") + "]";

    client.publish(mqttSensorLogTopic, message.c_str());
    Serial.printf("[MQTT] Sensor log sent: Cursor %d, %d rows, %d bytes.\n", cursor, sentCount, message.length());
  }
}

void mqttPublishResources() {
  if (!client.connected()) {
    mqttReconnect();
  }

  String payload = "{ \"free_heap\": " + String(ESP.getFreeHeap()) +
                   ", \"max_alloc_heap\": " + String(ESP.getMaxAllocHeap()) +
                   ", \"cpu_freq_mhz\": " + String(ESP.getCpuFreqMHz()) + " }";

  client.publish(mqttResourcesTopic, payload.c_str());
  Serial.print("[MQTT] Published payload: ");
  Serial.println(payload);
}

void mqttPublishTelemetry(time_t epochTime, float distance, int flow) {
  if (!client.connected()) {
    mqttReconnect();
  }

  String payloadStr = "{ \"timestamp\": " + String(epochTime) +
                      ", \"distance_cm\": " + String(distance) +
                      ", \"flow_raw\": " + String(flow) + " }";
  const uint8_t* payload = (uint8_t*) payloadStr.c_str();

  // QoS 1
  if (client.publish(mqttSensorTopic, payload, payloadStr.length(), false)) {
    Serial.print("[MQTT] Published payload: ");
    Serial.println(payloadStr);
  } else {
    Serial.println("[ERROR] Publish failed or timed out.");
  }
}

void mqttReconnect() {
  client.loop();

  while (!client.connected()) {
    Serial.println("[MQTT] Attempting MQTT connection...");

    if (client.connect(mqttSensorTopic)) {
      Serial.println("[MQTT] MQTT connected!");
      client.subscribe(mqttSensorLogTopic);
      client.subscribe(mqttTrainingTopic);
      Serial.printf("[MQTT] Subscribed to topics: %s, %s\n", mqttSensorLogTopic, mqttTrainingTopic);
    } else {
      Serial.printf("[ERROR] MQTT failed, rc=%d try again in 3 seconds.\n", client.state());
      delay(3000);
    }
  }
}

bool getAiToggleValue() {
  File file = SD.open(aiToggleDir, FILE_READ);

  if (!file) {
    Serial.println("[ERROR] Failed to open file, using 'false' as fallback.");
    return false;
  }

  String fileContent = "";

  while (file.available()) {
    fileContent += (char)file.read();
  }

  file.close();
  fileContent.trim();

  return fileContent.equals("true");
}

float getDistanceToWater() {
  digitalWrite(PIN_ULTRASONIC_TRIG, LOW);
  delayMicroseconds(2);
  digitalWrite(PIN_ULTRASONIC_TRIG, HIGH);
  delayMicroseconds(10);
  digitalWrite(PIN_ULTRASONIC_TRIG, LOW);

  long duration = pulseIn(PIN_ULTRASONIC_ECHO, HIGH, 30000);

  if (duration == 0) {
    return 0.0;
  }

  return (duration * 0.034) / 2.0;
}

time_t getTime() {
  struct tm timeinfo;

  if (getLocalTime(&timeinfo)) {
    return mktime(&timeinfo);
  }

  return 0; // Fallback
}

void logToSD(time_t epochTime, float distance, int flow) {
  struct tm timeinfo;

  File logFile = SD.open(sensorLogDir, FILE_APPEND);

  if (logFile) {
    char buffer[64];
    snprintf(buffer, sizeof(buffer), "%ld,%.2f,%d", epochTime, distance, flow);
    logFile.println(buffer);
    logFile.close();

    Serial.printf("[LOG] Timestamp: %ld | Distance: %.2f cm | Flow Raw: %d\n", epochTime, distance, flow);
  } else {
    Serial.println("[ERROR] Failed to open file.");
  }
}

void sendAlert(float distance, int flowRaw) {
  if (!getAiToggleValue()) {
    return;
  }

  std::vector<float> sample = {distance, (float)flowRaw};
  int cluster = kmeans.predict(sample);
  float distToCenter = kmeans.distanceToCentroid(sample);

  Serial.printf("[KMEANS] Cluster: %d | Distance to centroid: %.3f\n", cluster, distToCenter);

  if (distToCenter > ANOMALY_THRESHOLD) {
    Serial.println("[ALERT] Anomaly detected!");
    tone(PIN_BUZZER, 1000, 1000);
  }
}

void ensureFileExists(const char* path, const char* content) {
  if (SD.exists(path)) {
    Serial.printf("[FILE] %s already exists.\n", path);
    return;
  }

  File file = SD.open(path, FILE_WRITE);

  if (file) {
    if (content && strlen(content) > 0) {
      file.println(content);
    }

    file.close();
    Serial.printf("[FILE] Created %s.\n", path);
  } else {
    Serial.printf("[ERROR] Failed to create %s!\n", path);
  }
}

void initFiles() {
  ensureFileExists(aiToggleDir, "false");
  ensureFileExists(sensorLogDir, "Timestamp,Distance(cm),Flow(Raw)");
  ensureFileExists(trainingWindowDir, "0,0");
}

void initTime() {
  configTime(gmtOffset_sec, daylightOffset_sec, ntpServer);

  struct tm timeinfo;

  if (!getLocalTime(&timeinfo)) {
    Serial.println("[ERROR] Failed to obtain time from NTP. Retrying in 3 seconds");
    delay(3000);
  }

  Serial.println("[NTP] Time synchronized successfully!");
}

void applyTraining() {
  File configFile = SD.open(trainingWindowDir, FILE_READ);

  if (!configFile) {
    Serial.println("[INFO] No training window file found. Skipping training.");
    return;
  }

  long long start = 0;
  long long end = 0;
  sscanf(configFile.readString().c_str(), "%lld,%lld", &start, &end);
  configFile.close();

  if (start == 0 || end == 0) {
    Serial.println("[ERROR] Training window timestamps invalid or 0.");
    return;
  }

  time_t startEpoch = (time_t)start;
  time_t endEpoch = (time_t)end;

  Serial.printf("[KMEANS] Training window: %lld to %lld\n", (long long)startEpoch, (long long)endEpoch);
  trainFromLog(startEpoch, endEpoch);
}

void trainFromLog(time_t startEpoch, time_t endEpoch) {
  File logFile = SD.open(sensorLogDir, FILE_READ);

  if (!logFile) {
    Serial.println("[ERROR] Can't open log file for training.");
    return;
  }

  std::vector<std::vector<float>> dataset;
  logFile.readStringUntil('\n'); // Skip CSV header

  while (logFile.available()) {
    String line = logFile.readStringUntil('\n');
    if (line.length() < 10) continue;

    int firstComma = line.indexOf(',');
    int secondComma = line.indexOf(',', firstComma + 1);

    if (firstComma > 0 && secondComma > 0) {
      time_t ts = line.substring(0, firstComma).toInt();
      float distance = line.substring(firstComma + 1, secondComma).toFloat();
      float flow = line.substring(secondComma + 1).toFloat();

      if (ts >= startEpoch && ts <= endEpoch) {
        if (dataset.size() < MAX_TRAINING_LINES) {
          dataset.push_back({distance, flow});
        }
      }
    }
  }

  logFile.close();

  if (dataset.empty()) {
    Serial.println("[ERROR] No data found in the training window.");
    return;
  }

  kmeans.train(dataset);
  Serial.printf("[KMEANS] Trained successfully on %d samples ([Distance, Flow]).\n", dataset.size());
  kmeans.save(trainedModelDir);
}

