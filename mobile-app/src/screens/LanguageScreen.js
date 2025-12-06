import React, { useState } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  Dimensions,
  Image
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import AsyncStorage from '@react-native-async-storage/async-storage';

const { width, height } = Dimensions.get('window');

const LANGUAGES = [
  {
    code: 'vi',
    name: 'Tiếng Việt',
    flag: '🇻🇳',
    greeting: 'Xin chào!'
  },
  {
    code: 'en',
    name: 'English',
    flag: '🇺🇸',
    greeting: 'Hello!'
  },
  {
    code: 'km',
    name: 'ភាសាខ្មែរ',
    flag: '🇰🇭',
    greeting: 'សួស្តី!'
  }
];

export default function LanguageScreen({ navigation }) {
  const [selectedLanguage, setSelectedLanguage] = useState('vi');

  const handleLanguageSelect = async (languageCode) => {
    setSelectedLanguage(languageCode);
    await AsyncStorage.setItem('userLanguage', languageCode);
  };

  const handleContinue = () => {
    navigation.replace('Home');
  };

  return (
    <LinearGradient
      colors={['#FBD6E3', '#A9EDE9']}
      style={styles.container}
      start={{ x: 0, y: 0 }}
      end={{ x: 1, y: 1 }}
    >
      <View style={styles.content}>
        <Text style={styles.title}>Chọn ngôn ngữ / Language</Text>
        <Text style={styles.subtitle}>ជ្រើសរើសភាសា</Text>

        <View style={styles.languageContainer}>
          {LANGUAGES.map((lang) => (
            <TouchableOpacity
              key={lang.code}
              style={[
                styles.languageCard,
                selectedLanguage === lang.code && styles.languageCardSelected
              ]}
              onPress={() => handleLanguageSelect(lang.code)}
              activeOpacity={0.7}
            >
              <Text style={styles.flag}>{lang.flag}</Text>
              <Text style={styles.languageName}>{lang.name}</Text>
              <Text style={styles.greeting}>{lang.greeting}</Text>
            </TouchableOpacity>
          ))}
        </View>

        <TouchableOpacity
          style={styles.continueButton}
          onPress={handleContinue}
          activeOpacity={0.8}
        >
          <LinearGradient
            colors={['#FF8DAD', '#FF6B99']}
            style={styles.continueButtonGradient}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 0 }}
          >
            <Text style={styles.continueButtonText}>Tiếp tục / Continue</Text>
          </LinearGradient>
        </TouchableOpacity>
      </View>
    </LinearGradient>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1
  },
  content: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 20
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#FF8DAD',
    marginBottom: 5,
    textAlign: 'center'
  },
  subtitle: {
    fontSize: 18,
    color: '#FF6B99',
    marginBottom: 40,
    textAlign: 'center'
  },
  languageContainer: {
    width: '100%',
    marginBottom: 40
  },
  languageCard: {
    backgroundColor: 'rgba(255, 255, 255, 0.7)',
    borderRadius: 15,
    padding: 20,
    marginBottom: 15,
    flexDirection: 'row',
    alignItems: 'center',
    borderWidth: 2,
    borderColor: 'transparent',
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2
    },
    shadowOpacity: 0.25,
    shadowRadius: 3.84,
    elevation: 5
  },
  languageCardSelected: {
    borderColor: '#FF8DAD',
    backgroundColor: 'rgba(255, 141, 173, 0.1)'
  },
  flag: {
    fontSize: 40,
    marginRight: 15
  },
  languageName: {
    fontSize: 20,
    fontWeight: '600',
    color: '#FF6B99',
    flex: 1
  },
  greeting: {
    fontSize: 16,
    color: '#666'
  },
  continueButton: {
    width: '100%',
    borderRadius: 25,
    overflow: 'hidden'
  },
  continueButtonGradient: {
    paddingVertical: 15,
    alignItems: 'center'
  },
  continueButtonText: {
    color: '#FFF',
    fontSize: 18,
    fontWeight: 'bold'
  }
});
