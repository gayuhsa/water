
#include <PubSubClient.h>
#include <WiFi.h>
#include <cmath>
#include <cstdlib>
#include <time.h>
#include <vector>
#include "KMeans.h"

#define PIN_WATER_LEVEL 32
#define BUZZER_GPIO 14
#define PIN_FLOW_SENSOR 33

#define ANOMALY_THRESHOLD 0.1f
#define BOTTOM_MARGIN_MM 110
#define LOG_BUFFER_SIZE 2048
#define LOOP_DURATION_MS 5000
#define MAX_TRAINING_LINES 2000
#define MAX_LOG_LINES 2000
#define SIMULATE_FLOW 0
#define WATER_LEVEL_SENSOR_HEIGHT 50
#define FLOW_PULSE_NORMALIZATION 100.0f

const char* ssid = "";
const char* password = "";

const char* mqttServer = "broker.hivemq.com";
const char* mqttAiToggleTopic = "esp32/water/ai_toggle";
const char* mqttResourcesTopic = "esp32/water/resources";
const char* mqttSensorLogTopic = "esp32/water/sensor_log";
const char* mqttSensorTopic = "esp32/water/sensors";
const char* mqttTrainingTopic = "esp32/water/training_window";

const char* ntpServer = "pool.ntp.org";
const long gmtOffset_sec = 0;
const int daylightOffset_sec = 0;

WiFiClient espClient;
PubSubClient client(espClient);

KMeans kmeans(3, 100, 1e-4f);

struct SensorLog {
  time_t timestamp;
  int waterLevel;
  int flowPulses;
};

std::vector<SensorLog> sensorLogs;

bool isAiEnabled = true;

long long trainingStart = 0;
long long trainingEnd = 0;

volatile uint32_t pulseCount = 0;
unsigned long previousFlowMillis = 0;

void IRAM_ATTR countPulse() {
  pulseCount++;
}

std::vector<float> normalizeSample(int waterLevel, int flowPulses) {
  float normalizedWater = (float)waterLevel / (float)(BOTTOM_MARGIN_MM + WATER_LEVEL_SENSOR_HEIGHT);
  float normalizedFlow = (float)flowPulses / FLOW_PULSE_NORMALIZATION;

  return { normalizedWater, normalizedFlow };
}

void setup() {
  Serial.begin(115200);

#if !SIMULATE_FLOW
  previousFlowMillis = millis();
  pinMode(PIN_FLOW_SENSOR, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(PIN_FLOW_SENSOR), countPulse, FALLING);
#endif

  pinMode(PIN_WATER_LEVEL, INPUT);
  pinMode(BUZZER_GPIO, OUTPUT);

  analogReadResolution(12);
  analogSetPinAttenuation(PIN_WATER_LEVEL, ADC_11db);
  trainFromSampleDataset();

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
  int waterLevel = getWaterLevel();
  int flowPulses = getWaterFlow();

  sendAlert(waterLevel, flowPulses);
  logToMemory(epochTime, waterLevel, flowPulses);
  mqttPublishResources();
  mqttPublishTelemetry(epochTime, waterLevel, flowPulses);
}

void mqttCallback(char* topic, byte* payload, unsigned int length) {
  String message = "";

  for (unsigned int i = 0; i < length; i++) {
    message += (char)payload[i];
  }

  message.trim();
  Serial.printf("[MQTT] Message arrived on [%s]: %s\n", topic, message.c_str());

  if (String(topic) == mqttAiToggleTopic) {
    message.toLowerCase();
    isAiEnabled = (message == "true" || message == "1");

    Serial.printf("[KMEANS] Toggle status: %s\n", isAiEnabled);
  } else if (String(topic) == mqttTrainingTopic) {
    long long start = 0;
    long long end = 0;

    if (sscanf(message.c_str(), "%lld,%lld", &start, &end) == 2 && start > 0 && end > start) {
      trainingStart = start;
      trainingEnd = end;

      Serial.println("[KMEANS] Training window updated from MQTT!");

      if (getAiToggleValue()) {
        Serial.println("[KMEANS] Retraining with the new training window...");
        applyTraining();
      }
    } else {
      Serial.println("[ERROR] Invalid training window.");
    }
  } else if (String(topic) == mqttSensorLogTopic) {
    int cursor = message.toInt();
    if (cursor <= 0) cursor = 1;

    const int LINES_PER_PAGE = 50;
    int startIndex = (cursor - 1) * LINES_PER_PAGE;

    int sentCount = 0;
    bool hasMore = false;
    String payloadStr = "";

    String startWrapper = "[LOG_START: cursor=" + String(cursor) + "]\n";
    int metadataOverhead = startWrapper.length() + 100;

    for (int i = startIndex; i < (int)sensorLogs.size(); i++) {
      if (sentCount >= LINES_PER_PAGE) {
        hasMore = true;
        break;
      }

      const SensorLog& log = sensorLogs[i];

      String line = String((long)log.timestamp) + "," + String(log.waterLevel) + "," + String(log.flowPulses) + "\n";

      if (startWrapper.length() + payloadStr.length() + line.length() + metadataOverhead > LOG_BUFFER_SIZE) {
        hasMore = true;
        break;
      }

      payloadStr += line;
      sentCount++;
    }

    String response = startWrapper + payloadStr + "[LOG_END: cursor=" + String(cursor) + ", has_more=" + (hasMore ? "true" : "false") + "]";

    client.publish(mqttSensorLogTopic, response.c_str());

    Serial.printf("[MQTT] Sensor log sent: Cursor %d, %d rows, %d bytes.\n",
                  cursor, sentCount, response.length());
  }
}

void mqttPublishResources() {
  if (!client.connected()) {
    mqttReconnect();
  }

  String payload = "{ \"free_heap\": " + String(ESP.getFreeHeap()) + ", \"max_alloc_heap\": " + String(ESP.getMaxAllocHeap()) + ", \"cpu_freq_mhz\": " + String(ESP.getCpuFreqMHz()) + ", \"stored_logs\": " + String(sensorLogs.size()) + " }";

  client.publish(mqttResourcesTopic, payload.c_str());
  Serial.print("[MQTT] Published payload: ");
  Serial.println(payload);
}

void mqttPublishTelemetry(time_t epochTime, int waterLevel, int flowPulses) {
  if (!client.connected()) {
    mqttReconnect();
  }

  String payloadStr = "{ \"timestamp\": " + String(epochTime) + ", \"water_level_mm\": " + String(waterLevel) + ", \"flow_pulses\": " + String(flowPulses) + " }";

  const uint8_t* payload = (uint8_t*)payloadStr.c_str();

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
      client.subscribe(mqttAiToggleTopic);

      Serial.printf("[MQTT] Subscribed to topics: %s, %s, %s\n",
                    mqttSensorLogTopic, mqttTrainingTopic, mqttAiToggleTopic);
    } else {
      Serial.printf("[ERROR] MQTT failed, rc=%d try again in 3 seconds.\n", client.state());
      delay(3000);
    }
  }
}

bool getAiToggleValue() {
  return isAiEnabled;
}

int getWaterFlow() {
#if SIMULATE_FLOW
  static unsigned long lastSimulationMillis = 0;
  static uint32_t simulatedPulses = 0;

  unsigned long currentMillis = millis();
  unsigned long elapsedMillis = currentMillis - lastSimulationMillis;

  if (elapsedMillis == 0) {
    return 0;
  }

  // Simulate raw pulses per second
  float pulsesPerSecond = 25.0f;

  simulatedPulses += (uint32_t)roundf(
    pulsesPerSecond * elapsedMillis / 1000.0f
  );

  lastSimulationMillis = currentMillis;

  uint32_t pulses = simulatedPulses;
  simulatedPulses = 0;

  return (int)pulses;
#else
  unsigned long currentMillis = millis();
  unsigned long elapsedMillis = currentMillis - previousFlowMillis;

  if (elapsedMillis == 0) {
    return 0;
  }

  noInterrupts();
  uint32_t pulses = pulseCount;
  pulseCount = 0;
  interrupts();

  previousFlowMillis = currentMillis;

  return (int)pulses;
#endif
}

int getWaterLevel() {
  // Multiply by 50 millimeter
  int measured = (analogRead(PIN_WATER_LEVEL) * WATER_LEVEL_SENSOR_HEIGHT) / 4095;
  return measured + BOTTOM_MARGIN_MM;
}

time_t getTime() {
  struct tm timeinfo;

  if (getLocalTime(&timeinfo)) {
    return mktime(&timeinfo);
  }

  return 0;
}

void logToMemory(time_t epochTime, int waterLevel, int flowPulses) {
  if (sensorLogs.size() >= MAX_LOG_LINES) {
    sensorLogs.erase(sensorLogs.begin());
  }

  sensorLogs.push_back({ epochTime, waterLevel, flowPulses });

  Serial.printf("[LOG] Timestamp: %ld | Water Level: %d | Flow Pulses: %d\n",
                (long)epochTime, waterLevel, flowPulses);
}

void sendAlert(int waterLevel, int flowPulses) {
  if (!getAiToggleValue()) {
    return;
  }

  std::vector<float> sample = normalizeSample((float)waterLevel, (float)flowPulses);
  int cluster = kmeans.predict(sample);

  if (cluster < 0) {
    return;
  }

  float distToCenter = kmeans.distanceToCentroid(sample);

  Serial.printf("[KMEANS] Cluster: %d | Distance to centroid: %.3f\n", cluster, distToCenter);

  if (distToCenter > ANOMALY_THRESHOLD) {
    Serial.println("[ALERT] Anomaly detected!");
    tone(BUZZER_GPIO, 1000, 1000);
  }
}

void initTime() {
  configTime(gmtOffset_sec, daylightOffset_sec, ntpServer);

  struct tm timeinfo;

  if (!getLocalTime(&timeinfo)) {
    Serial.println("[ERROR] Failed to obtain time from NTP. Retrying in 3 seconds");
    delay(3000);
    return;
  }

  Serial.println("[NTP] Time synchronized successfully!");
}

void applyTraining() {
  if (trainingStart == 0 || trainingEnd == 0 || trainingStart >= trainingEnd) {
    Serial.println("[ERROR] Training window timestamps invalid or 0.");
    return;
  }

  time_t startEpoch = (time_t)trainingStart;
  time_t endEpoch = (time_t)trainingEnd;

  Serial.printf("[KMEANS] Training window: %lld to %lld\n",
                (long long)startEpoch, (long long)endEpoch);

  trainFromLog(startEpoch, endEpoch);
}

void trainFromLog(time_t startEpoch, time_t endEpoch) {
  std::vector<std::vector<float>> dataset;

  for (const auto& log : sensorLogs) {
    if (log.timestamp >= startEpoch && log.timestamp <= endEpoch) {
      if (dataset.size() < MAX_TRAINING_LINES) {
        dataset.push_back(normalizeSample((float)log.waterLevel, (float)log.flowPulses));
      }
    }
  }

  if (dataset.size() < 3) {
    Serial.println("[ERROR] Not enough data in the training window.");
    return;
  }

  kmeans.train(dataset);
  Serial.printf("[KMEANS] Trained successfully on %d samples ([Water Level, Flow Pulses]).\n",
                (int)dataset.size());
}

void trainFromSampleDataset() {
  std::vector<std::vector<float>> dataset = {
    // Normal operating conditions
    normalizeSample(112, 75),
    normalizeSample(115, 112),
    normalizeSample(118, 112),
    normalizeSample(120, 150),
    normalizeSample(123, 150),
    normalizeSample(125, 187),
    normalizeSample(128, 187),
    normalizeSample(130, 225),
    normalizeSample(132, 225),

    // Mid operating conditions
    normalizeSample(135, 300),
    normalizeSample(138, 337),
    normalizeSample(140, 337),
    normalizeSample(142, 375),
    normalizeSample(145, 375),
    normalizeSample(148, 412),

    // Higher operating conditions
    normalizeSample(150, 487),
    normalizeSample(153, 525),
    normalizeSample(155, 562),
    normalizeSample(158, 600),
    normalizeSample(160, 637)
  };

  kmeans.train(dataset);

  Serial.printf(
    "[KMEANS] Embedded dataset training completed: %d samples.\n",
    (int)dataset.size());
}

