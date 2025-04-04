import React, { useState } from 'react';
import { 
  View, 
  TextInput, 
  Button, 
  Text, 
  ActivityIndicator, 
  ScrollView, 
  Dimensions, 
  TouchableOpacity, 
  Platform, 
  StatusBar 
} from 'react-native';
import { LineChart, PieChart } from 'react-native-chart-kit';
import { WebView } from 'react-native-webview';
import { MaterialIcons } from '@expo/vector-icons';

interface PredictionData {
  dates: string[];
  prices: string[];
  trends: string[];
}

const generateChartHTML = (data: PredictionData | null, isDarkMode: boolean) => {
  if (!data) return '';
  return `
  <!DOCTYPE html>
  <html>
    <head>
      <meta charset="UTF-8" />
      <style>
        html, body, #main {
          margin: 0;
          padding: 0;
          height: 100%;
          background-color: ${isDarkMode ? '#1a1a1a' : '#fff'};
        }
      </style>
    </head>
    <body>
      <div id="main" style="width:100%; height:100%;"></div>
      <script src="https://cdn.jsdelivr.net/npm/echarts@5"></script>
      <script>
        var chart = echarts.init(document.getElementById('main'));
        chart.setOption({
          tooltip: { trigger: 'axis' },
          xAxis: {
            type: 'category',
            data: ${JSON.stringify(data.dates)},
            axisLabel: { 
              rotate: 45,
              color: '${isDarkMode ? '#fff' : '#000'}'
            }
          },
          yAxis: { 
            type: 'value',
            axisLabel: {
              color: '${isDarkMode ? '#fff' : '#000'}'
            }
          },
          series: [{
            data: ${JSON.stringify(data.prices.map(p => parseFloat(p)))},
            type: 'line',
            smooth: true,
            lineStyle: { color: '${isDarkMode ? '#82d1ff' : 'blue'}' }
          }]
        });
      </script>
    </body>
  </html>`;
};

export default function PredictScreen() {
  const [isDarkMode, setIsDarkMode] = useState(false);
  const [numDays, setNumDays] = useState("20");
  const [data, setData] = useState<PredictionData | null>(null);
  const [loading, setLoading] = useState(false);

  const colors = {
    background: isDarkMode ? '#121212' : '#f5f5f5',
    text: isDarkMode ? '#ffffff' : '#2d4059',
    cardBackground: isDarkMode ? '#1e1e1e' : '#ffffff',
    border: isDarkMode ? '#333333' : '#e0e0e0',
    primary: isDarkMode ? '#82d1ff' : '#2d4059',
    up: isDarkMode ? '#81C784' : '#4CAF50',
    down: isDarkMode ? '#E57373' : '#F44336'
  };

  const handleSubmit = async () => {
    setLoading(true);
    try {
      const resp = await fetch("http://192.168.68.100:5000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ days: numDays })
      });
      const json = await resp.json();
      setData(json);
    } catch (err) {
      console.error(err);
    }
    setLoading(false);
  };

  const getLabels = (dates: string[]) => {
    if (dates.length <= 7) return dates;
    const step = Math.ceil(dates.length / 7);
    return dates.map((d, i) => (i % step === 0 ? d : ""));
  };

  const chartWidth = data ? Math.max(Dimensions.get("window").width, data.dates.length * 50) : Dimensions.get("window").width;

  let upCount = 0, downCount = 0;
  let minPrice = 0, maxPrice = 0, avgPrice = 0;

  if (data) {
    const pricesNum = data.prices.map(p => parseFloat(p));
    upCount = data.trends.filter(t => t === "Up").length;
    downCount = data.trends.filter(t => t === "Down").length;
    minPrice = Math.min(...pricesNum);
    maxPrice = Math.max(...pricesNum);
    avgPrice = pricesNum.reduce((a, b) => a + b, 0) / pricesNum.length;
  }

  const pieData = [
    { name: "Up", count: upCount, color: colors.up, legendFontColor: colors.text, legendFontSize: 15 },
    { name: "Down", count: downCount, color: colors.down, legendFontColor: colors.text, legendFontSize: 15 }
  ];

  return (
    <ScrollView 
      contentContainerStyle={{ 
        marginTop: 20,
        padding: 40, 
        backgroundColor: colors.background,
        minHeight: '100%'
      }}
    >
      <StatusBar barStyle={isDarkMode ? 'light-content' : 'dark-content'} />

      <TouchableOpacity 
        onPress={() => setIsDarkMode(!isDarkMode)}
        style={{
          position: 'absolute',
          right: 20,
          top: Platform.OS === 'ios' ? 20 : 20,
          zIndex: 1
        }}
      >
        <MaterialIcons 
          name={isDarkMode ? 'brightness-7' : 'brightness-4'} 
          size={40} 
          color={colors.text} 
        />
      </TouchableOpacity>

      <Text style={{ 
        fontSize: 16, 
        color: colors.text, 
        marginBottom: 5, 
        fontWeight: '600' 
      }}>
        Enter number of days (1-365):
      </Text>

      <TextInput
        style={{ 
          borderWidth: 1,
          borderColor: colors.border,
          borderRadius: 8,
          marginVertical: 10,
          padding: 12,
          fontSize: 16,
          backgroundColor: colors.cardBackground,
          color: colors.text,
          shadowColor: '#000',
          shadowOffset: { width: 0, height: 2 },
          shadowOpacity: 0.1,
          shadowRadius: 4,
        }}
        keyboardType="numeric"
        value={numDays}
        onChangeText={setNumDays}
        placeholderTextColor={colors.text + '88'}
      />

      <Button 
        title="Predict" 
        onPress={handleSubmit} 
        color={colors.primary}
      />

      {loading && (
        <View style={{ height: 200, justifyContent: 'center' }}>
          <ActivityIndicator size="large" color={colors.primary} />
        </View>
      )}

      {data && (
        <>
          <Text style={{ 
            marginTop: 25, 
            fontSize: 18, 
            fontWeight: '700', 
            color: colors.text, 
            marginBottom: 10 
          }}>
            Predicted Prices
          </Text>
          
          <ScrollView horizontal showsHorizontalScrollIndicator={false}>
            <LineChart
              data={{
                labels: getLabels(data.dates),
                datasets: [{ data: data.prices.map(p => parseFloat(p)) }]
              }}
              width={chartWidth}
              height={250}
              chartConfig={{
                backgroundColor: colors.cardBackground,
                backgroundGradientFrom: colors.cardBackground,
                backgroundGradientTo: colors.cardBackground,
                decimalPlaces: 2,
                color: () => colors.primary,
                labelColor: () => colors.text,
                propsForDots: {
                  r: "4",
                  strokeWidth: "2",
                  stroke: colors.primary
                },
                propsForLabels: {
                  fontSize: 12
                },
                fillShadowGradient: colors.primary,
                fillShadowGradientOpacity: 0.2,
              }}
              bezier
              style={{
                marginVertical: 8,
                borderRadius: 16,
                paddingRight: 20,
                backgroundColor: colors.cardBackground,
                shadowColor: '#000',
                shadowOffset: { width: 0, height: 2 },
                shadowOpacity: 0.1,
                shadowRadius: 6,
              }}
            />
          </ScrollView>

          <Text style={{ 
            marginTop: 25, 
            fontSize: 18, 
            fontWeight: '700', 
            color: colors.text, 
            marginBottom: 10 
          }}>
            Trend Summary
          </Text>
          <View style={{
            borderRadius: 16,
            padding: 16,
            backgroundColor: colors.cardBackground,
            shadowColor: '#000',
            shadowOffset: { width: 0, height: 2 },
            shadowOpacity: 0.1,
            shadowRadius: 6,
          }}>
            <PieChart
              data={pieData}
              width={Dimensions.get("window").width - 40}
              height={220}
              chartConfig={{
                color: (opacity = 1) => colors.text,
                labelColor: () => colors.text,
              }}
              accessor="count"
              backgroundColor="transparent"
              paddingLeft="15"
              absolute
            />
          </View>

          <View style={{
            marginTop: 25,
            backgroundColor: colors.cardBackground,
            borderRadius: 12,
            padding: 16,
            shadowColor: '#000',
            shadowOffset: { width: 0, height: 2 },
            shadowOpacity: 0.1,
            shadowRadius: 6,
          }}>
            <Text style={{ 
              fontSize: 18, 
              fontWeight: '700', 
              color: colors.text, 
              marginBottom: 12 
            }}>
              Analysis
            </Text>
            <View style={{ flexDirection: 'row', flexWrap: 'wrap', justifyContent: 'space-between' }}>
              <View style={{ width: '48%', marginBottom: 10 }}>
                <Text style={{ color: '#777', fontSize: 14 }}>Start Date</Text>
                <Text style={{ fontSize: 16, fontWeight: '600', color: colors.text }}>{data.dates[0]}</Text>
              </View>
              <View style={{ width: '48%', marginBottom: 10 }}>
                <Text style={{ color: '#777', fontSize: 14 }}>End Date</Text>
                <Text style={{ fontSize: 16, fontWeight: '600', color: colors.text }}>{data.dates[data.dates.length - 1]}</Text>
              </View>
              <View style={{ width: '48%', marginBottom: 10 }}>
                <Text style={{ color: '#777', fontSize: 14 }}>Min Price</Text>
                <Text style={{ fontSize: 16, fontWeight: '600', color: colors.text }}>₹{minPrice.toFixed(2)}</Text>
              </View>
              <View style={{ width: '48%', marginBottom: 10 }}>
                <Text style={{ color: '#777', fontSize: 14 }}>Max Price</Text>
                <Text style={{ fontSize: 16, fontWeight: '600', color: colors.text }}>₹{maxPrice.toFixed(2)}</Text>
              </View>
              <View style={{ width: '100%' }}>
                <Text style={{ color: '#777', fontSize: 14 }}>Average Price</Text>
                <Text style={{ fontSize: 16, fontWeight: '600', color: colors.text }}>₹{avgPrice.toFixed(2)}</Text>
              </View>
            </View>
          </View>

          <Text style={{ 
            marginTop: 25, 
            fontSize: 18, 
            fontWeight: '700', 
            color: colors.text, 
            marginBottom: 10 
          }}>
            Interactive Price Chart
          </Text>
          <View style={{
            borderRadius: 12,
            overflow: 'hidden',
            shadowColor: '#000',
            shadowOffset: { width: 0, height: 2 },
            shadowOpacity: 0.1,
            shadowRadius: 6,
          }}>
            <WebView
              originWhitelist={['*']}
              source={{ html: generateChartHTML(data, isDarkMode) }}
              style={{
                height: 300,
                width: Dimensions.get("window").width - 40,
              }}
              javaScriptEnabled
              domStorageEnabled
            />
          </View>

          <View style={{
            marginTop: 25,
            backgroundColor: colors.cardBackground,
            borderRadius: 12,
            padding: 16,
            shadowColor: '#000',
            shadowOffset: { width: 0, height: 2 },
            shadowOpacity: 0.1,
            shadowRadius: 6,
          }}>
            <Text style={{ 
              fontSize: 18, 
              fontWeight: '700', 
              color: colors.text, 
              marginBottom: 12 
            }}>
              Daily Trends
            </Text>
            {data.dates.map((date, index) => (
              <View 
                key={index}
                style={{
                  flexDirection: 'row',
                  justifyContent: 'space-between',
                  paddingVertical: 8,
                  borderBottomWidth: index === data.dates.length - 1 ? 0 : 1,
                  borderBottomColor: colors.border,
                }}
              >
                <Text style={{ color: colors.text, flex: 2 }}>{date}</Text>
                <Text 
                  style={{ 
                    fontWeight: '600',
                    color: data.trends[index] === 'Up' ? colors.up : colors.down,
                    flex: 1,
                    textAlign: 'right'
                  }}
                >
                  {data.trends[index]}
                </Text>
              </View>
            ))}
          </View>
        </>
      )}
    </ScrollView>
  );
}