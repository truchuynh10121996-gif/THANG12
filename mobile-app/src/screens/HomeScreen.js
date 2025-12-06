import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  Dimensions,
  Image,
  ScrollView
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import AsyncStorage from '@react-native-async-storage/async-storage';

const { width, height } = Dimensions.get('window');

const TRANSLATIONS = {
  vi: {
    welcome: 'Xin chào!',
    subtitle: 'Tôi là Agribank Digital Guard',
    description: 'Trợ lý AI bảo vệ bạn khỏi lừa đảo',
    startChat: 'Bắt đầu trò chuyện',
    features: 'Tính năng',
    feature1: 'Phát hiện lừa đảo',
    feature2: 'Hỗ trợ 3 ngôn ngữ',
    feature3: 'Ghi âm giọng nói',
    feature4: 'Phát âm thanh',
    changeLanguage: 'Đổi ngôn ngữ'
  },
  en: {
    welcome: 'Welcome!',
    subtitle: 'I am Agribank Digital Guard',
    description: 'AI Assistant protecting you from fraud',
    startChat: 'Start Chatting',
    features: 'Features',
    feature1: 'Fraud Detection',
    feature2: 'Support 3 Languages',
    feature3: 'Voice Recording',
    feature4: 'Text-to-Speech',
    changeLanguage: 'Change Language'
  },
  km: {
    welcome: 'សូមស្វាគមន៍!',
    subtitle: 'ខ្ញុំគឺ Agribank Digital Guard',
    description: 'ជំនួយការ AI ការពារអ្នកពីការលួចបន្លំ',
    startChat: 'ចាប់ផ្តើមជជែក',
    features: 'លក្ខណៈពិសេស',
    feature1: 'រកឃើញការលួចបន្លំ',
    feature2: 'គាំទ្រ 3 ភាសា',
    feature3: 'ថតសំឡេង',
    feature4: 'អានសំឡេង',
    changeLanguage: 'ផ្លាស់ប្តូរភាសា'
  }
};

export default function HomeScreen({ navigation }) {
  const [language, setLanguage] = useState('vi');
  const [text, setText] = useState(TRANSLATIONS.vi);

  useEffect(() => {
    loadLanguage();
  }, []);

  const loadLanguage = async () => {
    const savedLanguage = await AsyncStorage.getItem('userLanguage');
    if (savedLanguage) {
      setLanguage(savedLanguage);
      setText(TRANSLATIONS[savedLanguage]);
    }
  };

  const handleStartChat = () => {
    navigation.navigate('Chat', { language });
  };

  const handleChangeLanguage = () => {
    navigation.navigate('Language');
  };

  return (
    <LinearGradient
      colors={['#FBD6E3', '#A9EDE9']}
      style={styles.container}
      start={{ x: 0, y: 0 }}
      end={{ x: 1, y: 1 }}
    >
      <ScrollView contentContainerStyle={styles.scrollContent}>
        <View style={styles.header}>
          <TouchableOpacity
            style={styles.languageButton}
            onPress={handleChangeLanguage}
          >
            <Ionicons name="language" size={24} color="#FF8DAD" />
            <Text style={styles.languageButtonText}>{text.changeLanguage}</Text>
          </TouchableOpacity>
        </View>

        <View style={styles.content}>
          <Image
            source={require('../../assets/logo-agribank.png')}
            style={styles.logo}
            resizeMode="contain"
          />

          <Text style={styles.welcome}>{text.welcome}</Text>
          <Text style={styles.subtitle}>{text.subtitle}</Text>
          <Text style={styles.description}>{text.description}</Text>

          <View style={styles.featuresContainer}>
            <Text style={styles.featuresTitle}>{text.features}</Text>

            <View style={styles.featuresGrid}>
              <View style={styles.featureCard}>
                <Ionicons name="shield-checkmark" size={40} color="#FF8DAD" />
                <Text style={styles.featureText}>{text.feature1}</Text>
              </View>

              <View style={styles.featureCard}>
                <Ionicons name="globe" size={40} color="#FF8DAD" />
                <Text style={styles.featureText}>{text.feature2}</Text>
              </View>

              <View style={styles.featureCard}>
                <Ionicons name="mic" size={40} color="#FF8DAD" />
                <Text style={styles.featureText}>{text.feature3}</Text>
              </View>

              <View style={styles.featureCard}>
                <Ionicons name="volume-high" size={40} color="#FF8DAD" />
                <Text style={styles.featureText}>{text.feature4}</Text>
              </View>
            </View>
          </View>

          <TouchableOpacity
            style={styles.startButton}
            onPress={handleStartChat}
            activeOpacity={0.8}
          >
            <LinearGradient
              colors={['#FF8DAD', '#FF6B99']}
              style={styles.startButtonGradient}
              start={{ x: 0, y: 0 }}
              end={{ x: 1, y: 0 }}
            >
              <Ionicons name="chatbubbles" size={24} color="#FFF" />
              <Text style={styles.startButtonText}>{text.startChat}</Text>
            </LinearGradient>
          </TouchableOpacity>
        </View>
      </ScrollView>
    </LinearGradient>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1
  },
  scrollContent: {
    flexGrow: 1
  },
  header: {
    paddingTop: 50,
    paddingHorizontal: 20,
    alignItems: 'flex-end'
  },
  languageButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(255, 255, 255, 0.7)',
    paddingHorizontal: 15,
    paddingVertical: 8,
    borderRadius: 20
  },
  languageButtonText: {
    marginLeft: 5,
    color: '#FF8DAD',
    fontWeight: '600'
  },
  content: {
    flex: 1,
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingBottom: 30
  },
  logo: {
    width: width * 0.4,
    height: width * 0.4,
    marginTop: 20
  },
  welcome: {
    fontSize: 32,
    fontWeight: 'bold',
    color: '#FF8DAD',
    marginTop: 20
  },
  subtitle: {
    fontSize: 20,
    fontWeight: '600',
    color: '#FF6B99',
    marginTop: 10,
    textAlign: 'center'
  },
  description: {
    fontSize: 16,
    color: '#4A4A4A',
    marginTop: 5,
    textAlign: 'center'
  },
  featuresContainer: {
    width: '100%',
    marginTop: 30
  },
  featuresTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#FF8DAD',
    marginBottom: 15
  },
  featuresGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between'
  },
  featureCard: {
    width: '48%',
    backgroundColor: 'rgba(255, 255, 255, 0.7)',
    borderRadius: 15,
    padding: 15,
    marginBottom: 15,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2
    },
    shadowOpacity: 0.25,
    shadowRadius: 3.84,
    elevation: 5
  },
  featureText: {
    marginTop: 10,
    fontSize: 14,
    color: '#FF6B99',
    textAlign: 'center',
    fontWeight: '600'
  },
  startButton: {
    width: '100%',
    borderRadius: 25,
    overflow: 'hidden',
    marginTop: 30
  },
  startButtonGradient: {
    flexDirection: 'row',
    paddingVertical: 15,
    alignItems: 'center',
    justifyContent: 'center'
  },
  startButtonText: {
    color: '#FFF',
    fontSize: 18,
    fontWeight: 'bold',
    marginLeft: 10
  }
});
