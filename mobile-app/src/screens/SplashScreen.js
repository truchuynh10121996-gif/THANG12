import React, { useEffect } from 'react';
import { View, Image, StyleSheet, Animated, Dimensions } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';

const { width, height } = Dimensions.get('window');

export default function SplashScreen({ navigation }) {
  const fadeAnim = new Animated.Value(0);
  const scaleAnim = new Animated.Value(0.5);

  useEffect(() => {
    // Animation
    Animated.parallel([
      Animated.timing(fadeAnim, {
        toValue: 1,
        duration: 1000,
        useNativeDriver: true
      }),
      Animated.spring(scaleAnim, {
        toValue: 1,
        friction: 4,
        tension: 40,
        useNativeDriver: true
      })
    ]).start();

    // Navigate to Language screen after 2.5 seconds
    const timer = setTimeout(() => {
      navigation.replace('Language');
    }, 2500);

    return () => clearTimeout(timer);
  }, []);

  return (
    <LinearGradient
      colors={['#FBD6E3', '#A9EDE9']}
      style={styles.container}
      start={{ x: 0, y: 0 }}
      end={{ x: 1, y: 1 }}
    >
      <Animated.View
        style={[
          styles.logoContainer,
          {
            opacity: fadeAnim,
            transform: [{ scale: scaleAnim }]
          }
        ]}
      >
        <Image
          source={require('../../assets/logo-agribank.png')}
          style={styles.logo}
          resizeMode="contain"
        />
      </Animated.View>

      <Animated.Text
        style={[
          styles.title,
          { opacity: fadeAnim }
        ]}
      >
        AGRIBANK DIGITAL GUARD
      </Animated.Text>

      <Animated.Text
        style={[
          styles.subtitle,
          { opacity: fadeAnim }
        ]}
      >
        🛡️ Bảo vệ bạn khỏi lừa đảo
      </Animated.Text>
    </LinearGradient>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center'
  },
  logoContainer: {
    marginBottom: 30,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 4
    },
    shadowOpacity: 0.3,
    shadowRadius: 4.65,
    elevation: 8
  },
  logo: {
    width: width * 0.5,
    height: width * 0.5
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#FF8DAD',
    marginTop: 20,
    textAlign: 'center',
    letterSpacing: 1
  },
  subtitle: {
    fontSize: 16,
    color: '#FF6B99',
    marginTop: 10,
    textAlign: 'center'
  }
});
