import React, { useState, useEffect, useRef } from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  KeyboardAvoidingView,
  Platform,
  ScrollView,
  ActivityIndicator,
  Alert,
  Modal,
  Image
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import * as ImagePicker from 'expo-image-picker';
import ChatBubble from '../components/ChatBubble';
import VoiceRecorder from '../components/VoiceRecorder';
import { sendMessage, synthesizeSpeech, analyzeImage } from '../services/api';

export default function ChatScreen({ route, navigation }) {
  const { language = 'vi' } = route.params || {};

  const [messages, setMessages] = useState([]);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [conversationId, setConversationId] = useState(null);

  // Image states
  const [selectedImage, setSelectedImage] = useState(null);
  const [showImageModal, setShowImageModal] = useState(false);
  const [isAnalyzingImage, setIsAnalyzingImage] = useState(false);

  const scrollViewRef = useRef();

  useEffect(() => {
    // Tin nhắn chào mừng
    const welcomeMessage = getWelcomeMessage(language);
    setMessages([
      {
        id: '1',
        text: welcomeMessage,
        sender: 'bot',
        timestamp: new Date()
      }
    ]);
  }, []);

  const getWelcomeMessage = (lang) => {
    const messages = {
      vi: `Xin chào! 👋\n\nTôi là Agribank Digital Guard, trợ lý AI chuyên về phòng chống lừa đảo.\n\nNếu bạn gặp bất kỳ tình huống đáng ngờ nào, hãy mô tả cho tôi. Tôi sẽ phân tích và đưa ra cảnh báo cùng hướng dẫn xử lý an toàn.\n\nBạn có thể nhập văn bản hoặc dùng nút 🎤 để ghi âm giọng nói.`,
      en: `Hello! 👋\n\nI am Agribank Digital Guard, an AI assistant specializing in fraud prevention.\n\nIf you encounter any suspicious situations, please describe them to me. I will analyze and provide warnings along with safe handling instructions.\n\nYou can type text or use the 🎤 button to record your voice.`,
      km: `សួស្តី! 👋\n\nខ្ញុំគឺ Agribank Digital Guard ជំនួយការ AI ឯកទេសខាងការពារការលួចបន្លំ។\n\nប្រសិនបើអ្នកជួបស្ថានភាពសង្ស័យ សូមពណ៌នាដល់ខ្ញុំ។ ខ្ញុំនឹងវិភាគ និងផ្តល់ការព្រមាន រួមជាមួយការណែនាំសុវត្ថិភាព។\n\nអ្នកអាចវាយអក្សរ ឬប្រើប៊ូតុង 🎤 ដើម្បីថតសំឡេង។`
    };
    return messages[lang] || messages.vi;
  };

  const handleSendMessage = async () => {
    if (!inputText.trim()) return;

    const userMessage = {
      id: Date.now().toString(),
      text: inputText,
      sender: 'user',
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputText('');
    setIsLoading(true);

    try {
      const response = await sendMessage({
        message: inputText,
        conversationId,
        language
      });

      if (!conversationId) {
        setConversationId(response.conversationId);
      }

      const botMessage = {
        id: (Date.now() + 1).toString(),
        text: response.response,
        sender: 'bot',
        timestamp: new Date(),
        isFraudAlert: response.isFraudAlert,
        audioData: null // Will be loaded on demand
      };

      setMessages(prev => [...prev, botMessage]);

    } catch (error) {
      console.error('Send message error:', error);
      Alert.alert('Lỗi', 'Không thể gửi tin nhắn. Vui lòng thử lại.');

      const errorMessage = {
        id: (Date.now() + 1).toString(),
        text: 'Xin lỗi, tôi gặp sự cố kết nối. Vui lòng thử lại sau.',
        sender: 'bot',
        timestamp: new Date()
      };

      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleVoiceRecorded = (transcription) => {
    if (transcription) {
      setInputText(transcription);
    }
  };

  // Image picker handlers
  const handlePickImage = async () => {
    // Request permission
    const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (status !== 'granted') {
      Alert.alert(
        language === 'vi' ? 'Cần quyền truy cập' : 'Permission Required',
        language === 'vi'
          ? 'Vui lòng cấp quyền truy cập thư viện ảnh để sử dụng tính năng này.'
          : 'Please grant photo library access to use this feature.'
      );
      return;
    }

    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: false,
      quality: 0.8,
      base64: true
    });

    if (!result.canceled && result.assets[0]) {
      setSelectedImage(result.assets[0]);
      setShowImageModal(true);
    }
  };

  const handleTakePhoto = async () => {
    // Request permission
    const { status } = await ImagePicker.requestCameraPermissionsAsync();
    if (status !== 'granted') {
      Alert.alert(
        language === 'vi' ? 'Cần quyền truy cập' : 'Permission Required',
        language === 'vi'
          ? 'Vui lòng cấp quyền truy cập camera để sử dụng tính năng này.'
          : 'Please grant camera access to use this feature.'
      );
      return;
    }

    const result = await ImagePicker.launchCameraAsync({
      allowsEditing: false,
      quality: 0.8,
      base64: true
    });

    if (!result.canceled && result.assets[0]) {
      setSelectedImage(result.assets[0]);
      setShowImageModal(true);
    }
  };

  const handleImageOptions = () => {
    Alert.alert(
      language === 'vi' ? 'Chọn ảnh' : 'Select Image',
      language === 'vi'
        ? 'Chọn cách lấy ảnh tin nhắn để kiểm tra lừa đảo'
        : 'Choose how to get the message screenshot to check for fraud',
      [
        {
          text: language === 'vi' ? 'Chụp ảnh' : 'Take Photo',
          onPress: handleTakePhoto
        },
        {
          text: language === 'vi' ? 'Chọn từ thư viện' : 'Choose from Library',
          onPress: handlePickImage
        },
        {
          text: language === 'vi' ? 'Hủy' : 'Cancel',
          style: 'cancel'
        }
      ]
    );
  };

  const handleAnalyzeImage = async () => {
    if (!selectedImage || !selectedImage.base64) {
      Alert.alert('Error', 'No image selected');
      return;
    }

    setShowImageModal(false);
    setIsAnalyzingImage(true);

    // Add user message with image
    const userMessage = {
      id: Date.now().toString(),
      text: language === 'vi' ? '[Ảnh chụp màn hình]' : '[Screenshot]',
      sender: 'user',
      timestamp: new Date(),
      hasImage: true,
      imageUri: selectedImage.uri
    };

    setMessages(prev => [...prev, userMessage]);

    try {
      const imageBase64 = `data:image/jpeg;base64,${selectedImage.base64}`;

      const response = await analyzeImage({
        imageBase64,
        conversationId,
        language
      });

      if (!conversationId && response.conversationId) {
        setConversationId(response.conversationId);
      }

      const botMessage = {
        id: (Date.now() + 1).toString(),
        text: response.response,
        sender: 'bot',
        timestamp: new Date(),
        isFraudAlert: response.isFraudAlert,
        ocrResult: response.ocrResult
      };

      setMessages(prev => [...prev, botMessage]);

      if (response.isFraudAlert) {
        Alert.alert(
          language === 'vi' ? '⚠️ Cảnh báo lừa đảo!' : '⚠️ Fraud Alert!',
          language === 'vi'
            ? 'Phát hiện dấu hiệu lừa đảo trong ảnh. Vui lòng đọc phân tích chi tiết.'
            : 'Fraud indicators detected in the image. Please read the detailed analysis.'
        );
      }

    } catch (error) {
      console.error('Analyze image error:', error);
      Alert.alert(
        language === 'vi' ? 'Lỗi' : 'Error',
        language === 'vi'
          ? 'Không thể phân tích ảnh. Vui lòng thử lại.'
          : 'Cannot analyze image. Please try again.'
      );

      const errorMessage = {
        id: (Date.now() + 1).toString(),
        text: language === 'vi'
          ? 'Xin lỗi, tôi không thể phân tích ảnh này. Vui lòng thử lại hoặc mô tả nội dung tin nhắn bằng văn bản.'
          : 'Sorry, I cannot analyze this image. Please try again or describe the message content in text.',
        sender: 'bot',
        timestamp: new Date()
      };

      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsAnalyzingImage(false);
      setSelectedImage(null);
    }
  };

  const scrollToBottom = () => {
    setTimeout(() => {
      scrollViewRef.current?.scrollToEnd({ animated: true });
    }, 100);
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  return (
    <LinearGradient
      colors={['#FBD6E3', '#A9EDE9']}
      style={styles.container}
      start={{ x: 0, y: 0 }}
      end={{ x: 1, y: 1 }}
    >
      <KeyboardAvoidingView
        style={styles.container}
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        keyboardVerticalOffset={Platform.OS === 'ios' ? 0 : 20}
      >
        {/* Header */}
        <View style={styles.header}>
          <TouchableOpacity
            onPress={() => navigation.goBack()}
            style={styles.backButton}
          >
            <Ionicons name="arrow-back" size={24} color="#FF8DAD" />
          </TouchableOpacity>

          <View style={styles.headerTitleContainer}>
            <Text style={styles.headerTitle}>Digital Guard</Text>
            <Text style={styles.headerSubtitle}>🛡️ AI Assistant</Text>
          </View>

          <TouchableOpacity style={styles.infoButton}>
            <Ionicons name="information-circle" size={24} color="#FF8DAD" />
          </TouchableOpacity>
        </View>

        {/* Messages */}
        <ScrollView
          ref={scrollViewRef}
          style={styles.messagesContainer}
          contentContainerStyle={styles.messagesContent}
          showsVerticalScrollIndicator={false}
        >
          {messages.map((message) => (
            <ChatBubble
              key={message.id}
              message={message}
              language={language}
            />
          ))}

          {(isLoading || isAnalyzingImage) && (
            <View style={styles.loadingContainer}>
              <ActivityIndicator size="small" color="#FF8DAD" />
              <Text style={styles.loadingText}>
                {isAnalyzingImage
                  ? (language === 'vi' ? 'Đang phân tích ảnh...' : 'Analyzing image...')
                  : (language === 'vi' ? 'Đang suy nghĩ...' : 'Thinking...')
                }
              </Text>
            </View>
          )}
        </ScrollView>

        {/* Input */}
        <View style={styles.inputContainer}>
          <View style={styles.inputWrapper}>
            {/* Camera/Image Button */}
            <TouchableOpacity
              style={styles.cameraButton}
              onPress={handleImageOptions}
              disabled={isLoading || isAnalyzingImage}
            >
              <Ionicons
                name="camera"
                size={24}
                color={isLoading || isAnalyzingImage ? '#CCC' : '#FF8DAD'}
              />
            </TouchableOpacity>

            <TextInput
              style={styles.input}
              placeholder={
                language === 'vi' ? 'Nhập tin nhắn...' :
                language === 'en' ? 'Type a message...' :
                'វាយសារ...'
              }
              value={inputText}
              onChangeText={setInputText}
              multiline
              maxLength={500}
              editable={!isLoading && !isAnalyzingImage}
            />

            <VoiceRecorder
              language={language}
              onTranscriptionComplete={handleVoiceRecorded}
            />

            <TouchableOpacity
              style={[
                styles.sendButton,
                !inputText.trim() && styles.sendButtonDisabled
              ]}
              onPress={handleSendMessage}
              disabled={!inputText.trim() || isLoading || isAnalyzingImage}
            >
              <Ionicons
                name="send"
                size={24}
                color={inputText.trim() ? '#FF8DAD' : '#CCC'}
              />
            </TouchableOpacity>
          </View>
        </View>
      </KeyboardAvoidingView>

      {/* Image Preview Modal */}
      <Modal
        visible={showImageModal}
        transparent={true}
        animationType="slide"
        onRequestClose={() => setShowImageModal(false)}
      >
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <View style={styles.modalHeader}>
              <Text style={styles.modalTitle}>
                {language === 'vi' ? 'Xem trước ảnh' : 'Image Preview'}
              </Text>
              <TouchableOpacity
                onPress={() => {
                  setShowImageModal(false);
                  setSelectedImage(null);
                }}
              >
                <Ionicons name="close" size={24} color="#666" />
              </TouchableOpacity>
            </View>

            {selectedImage && (
              <Image
                source={{ uri: selectedImage.uri }}
                style={styles.previewImage}
                resizeMode="contain"
              />
            )}

            <Text style={styles.modalHint}>
              {language === 'vi'
                ? 'Gửi ảnh này để phân tích nội dung và kiểm tra dấu hiệu lừa đảo'
                : 'Send this image to analyze content and check for fraud indicators'}
            </Text>

            <View style={styles.modalButtons}>
              <TouchableOpacity
                style={styles.cancelButton}
                onPress={() => {
                  setShowImageModal(false);
                  setSelectedImage(null);
                }}
              >
                <Text style={styles.cancelButtonText}>
                  {language === 'vi' ? 'Hủy' : 'Cancel'}
                </Text>
              </TouchableOpacity>

              <TouchableOpacity
                style={styles.analyzeButton}
                onPress={handleAnalyzeImage}
              >
                <Text style={styles.analyzeButtonText}>
                  {language === 'vi' ? 'Gửi phân tích' : 'Analyze'}
                </Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>
    </LinearGradient>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingTop: 50,
    paddingBottom: 15,
    paddingHorizontal: 20,
    backgroundColor: 'rgba(255, 255, 255, 0.5)'
  },
  backButton: {
    padding: 5
  },
  headerTitleContainer: {
    flex: 1,
    alignItems: 'center'
  },
  headerTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#FF8DAD'
  },
  headerSubtitle: {
    fontSize: 12,
    color: '#FF6B99'
  },
  infoButton: {
    padding: 5
  },
  messagesContainer: {
    flex: 1
  },
  messagesContent: {
    padding: 15,
    paddingBottom: 10
  },
  loadingContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    alignSelf: 'flex-start',
    backgroundColor: 'rgba(255, 255, 255, 0.7)',
    borderRadius: 20,
    paddingHorizontal: 15,
    paddingVertical: 10,
    marginTop: 10
  },
  loadingText: {
    marginLeft: 10,
    color: '#FF8DAD',
    fontSize: 14
  },
  inputContainer: {
    paddingHorizontal: 15,
    paddingVertical: 10,
    backgroundColor: 'rgba(255, 255, 255, 0.5)'
  },
  inputWrapper: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    backgroundColor: '#FFF',
    borderRadius: 25,
    paddingHorizontal: 15,
    paddingVertical: 8,
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2
    },
    shadowOpacity: 0.25,
    shadowRadius: 3.84,
    elevation: 5
  },
  input: {
    flex: 1,
    maxHeight: 100,
    fontSize: 16,
    color: '#333',
    paddingVertical: 5
  },
  sendButton: {
    marginLeft: 10,
    padding: 5
  },
  sendButtonDisabled: {
    opacity: 0.5
  },
  cameraButton: {
    marginRight: 10,
    padding: 5
  },
  // Modal styles
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    justifyContent: 'center',
    alignItems: 'center'
  },
  modalContent: {
    backgroundColor: '#FFF',
    borderRadius: 16,
    padding: 20,
    width: '90%',
    maxHeight: '80%'
  },
  modalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 15
  },
  modalTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#FF8DAD'
  },
  previewImage: {
    width: '100%',
    height: 300,
    borderRadius: 8,
    marginBottom: 15
  },
  modalHint: {
    textAlign: 'center',
    color: '#666',
    fontSize: 14,
    marginBottom: 20
  },
  modalButtons: {
    flexDirection: 'row',
    justifyContent: 'space-between'
  },
  cancelButton: {
    flex: 1,
    padding: 15,
    borderRadius: 8,
    backgroundColor: '#F5F5F5',
    marginRight: 10,
    alignItems: 'center'
  },
  cancelButtonText: {
    color: '#666',
    fontSize: 16,
    fontWeight: '600'
  },
  analyzeButton: {
    flex: 1,
    padding: 15,
    borderRadius: 8,
    backgroundColor: '#FF8DAD',
    marginLeft: 10,
    alignItems: 'center'
  },
  analyzeButtonText: {
    color: '#FFF',
    fontSize: 16,
    fontWeight: '600'
  }
});
