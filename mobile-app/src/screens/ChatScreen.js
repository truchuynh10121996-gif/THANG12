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
  Image,
  Modal
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';
import * as ImagePicker from 'expo-image-picker';
import ChatBubble from '../components/ChatBubble';
import VoiceRecorder from '../components/VoiceRecorder';
import { sendMessage, synthesizeSpeech, analyzeImageForFraud } from '../services/api';

export default function ChatScreen({ route, navigation }) {
  const { language = 'vi' } = route.params || {};

  const [messages, setMessages] = useState([]);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [conversationId, setConversationId] = useState(null);
  const [selectedImage, setSelectedImage] = useState(null);
  const [showImageOptions, setShowImageOptions] = useState(false);

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

  // Xin quyền truy cập camera và thư viện ảnh
  const requestPermissions = async () => {
    const { status: cameraStatus } = await ImagePicker.requestCameraPermissionsAsync();
    const { status: libraryStatus } = await ImagePicker.requestMediaLibraryPermissionsAsync();

    if (cameraStatus !== 'granted' || libraryStatus !== 'granted') {
      Alert.alert(
        language === 'vi' ? 'Cần quyền truy cập' : 'Permission Required',
        language === 'vi'
          ? 'Vui lòng cấp quyền truy cập camera và thư viện ảnh để sử dụng tính năng này.'
          : 'Please grant camera and photo library access to use this feature.'
      );
      return false;
    }
    return true;
  };

  // Chụp ảnh từ camera
  const takePhoto = async () => {
    setShowImageOptions(false);

    const hasPermission = await requestPermissions();
    if (!hasPermission) return;

    const result = await ImagePicker.launchCameraAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      quality: 0.8
    });

    if (!result.canceled && result.assets[0]) {
      setSelectedImage(result.assets[0].uri);
    }
  };

  // Chọn ảnh từ thư viện
  const pickImage = async () => {
    setShowImageOptions(false);

    const hasPermission = await requestPermissions();
    if (!hasPermission) return;

    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      quality: 0.8
    });

    if (!result.canceled && result.assets[0]) {
      setSelectedImage(result.assets[0].uri);
    }
  };

  // Xóa ảnh đã chọn
  const removeSelectedImage = () => {
    setSelectedImage(null);
  };

  // Gửi ảnh để phân tích
  const handleSendImage = async () => {
    if (!selectedImage || isLoading) return;

    const imageUri = selectedImage;

    // Thêm tin nhắn người dùng với ảnh
    const userMessage = {
      id: Date.now().toString(),
      text: language === 'vi' ? 'Đã tải ảnh lên để phân tích' : 'Image uploaded for analysis',
      sender: 'user',
      timestamp: new Date(),
      imageUrl: imageUri
    };

    setMessages(prev => [...prev, userMessage]);
    setSelectedImage(null);
    setIsLoading(true);

    try {
      const response = await analyzeImageForFraud({
        imageUri,
        language,
        conversationId
      });

      if (!conversationId) {
        setConversationId(response.conversationId);
      }

      // Tạo tin nhắn bot với kết quả phân tích
      let botText = response.response;

      // Thêm thông tin văn bản trích xuất nếu có
      if (response.extractedText && response.extractedText.length > 0) {
        const extractedLabel = language === 'vi' ? 'Văn bản trích xuất từ ảnh' : 'Text extracted from image';
        const shortText = response.extractedText.length > 150
          ? response.extractedText.substring(0, 150) + '...'
          : response.extractedText;
        botText = `📝 ${extractedLabel}:\n"${shortText}"\n\n---\n\n${botText}`;
      }

      const botMessage = {
        id: (Date.now() + 1).toString(),
        text: botText,
        sender: 'bot',
        timestamp: new Date(),
        isFraudAlert: response.isFraudAlert,
        analysis: response.analysis
      };

      setMessages(prev => [...prev, botMessage]);

      if (response.isFraudAlert) {
        Alert.alert(
          '⚠️ ' + (language === 'vi' ? 'Cảnh báo lừa đảo!' : 'Fraud Alert!'),
          language === 'vi'
            ? 'Phát hiện dấu hiệu lừa đảo trong ảnh. Vui lòng xem chi tiết phản hồi.'
            : 'Fraud indicators detected in the image. Please review the response.'
        );
      }

    } catch (error) {
      console.error('Send image error:', error);
      Alert.alert(
        language === 'vi' ? 'Lỗi' : 'Error',
        language === 'vi' ? 'Không thể phân tích ảnh. Vui lòng thử lại.' : 'Could not analyze image. Please try again.'
      );

      const errorMessage = {
        id: (Date.now() + 1).toString(),
        text: language === 'vi'
          ? 'Xin lỗi, không thể phân tích ảnh. Vui lòng thử lại.'
          : 'Sorry, could not analyze the image. Please try again.',
        sender: 'bot',
        timestamp: new Date()
      };

      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
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

          {isLoading && (
            <View style={styles.loadingContainer}>
              <ActivityIndicator size="small" color="#FF8DAD" />
              <Text style={styles.loadingText}>Đang suy nghĩ...</Text>
            </View>
          )}
        </ScrollView>

        {/* Image Preview */}
        {selectedImage && (
          <View style={styles.imagePreviewContainer}>
            <Image source={{ uri: selectedImage }} style={styles.imagePreview} />
            <View style={styles.imagePreviewInfo}>
              <Text style={styles.imagePreviewText}>
                {language === 'vi' ? 'Ảnh đã chọn' : 'Image selected'}
              </Text>
              <TouchableOpacity onPress={removeSelectedImage} style={styles.removeImageBtn}>
                <Ionicons name="close-circle" size={24} color="#FF6B99" />
              </TouchableOpacity>
            </View>
          </View>
        )}

        {/* Input */}
        <View style={styles.inputContainer}>
          <View style={styles.inputWrapper}>
            {/* Camera Button */}
            <TouchableOpacity
              style={styles.cameraButton}
              onPress={() => setShowImageOptions(true)}
              disabled={isLoading}
            >
              <Ionicons
                name="camera"
                size={24}
                color={isLoading ? '#CCC' : '#FF8DAD'}
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
              editable={!selectedImage}
            />

            <VoiceRecorder
              language={language}
              onTranscriptionComplete={handleVoiceRecorded}
            />

            <TouchableOpacity
              style={[
                styles.sendButton,
                (!inputText.trim() && !selectedImage) && styles.sendButtonDisabled
              ]}
              onPress={selectedImage ? handleSendImage : handleSendMessage}
              disabled={(!inputText.trim() && !selectedImage) || isLoading}
            >
              <Ionicons
                name={selectedImage ? 'image' : 'send'}
                size={24}
                color={(inputText.trim() || selectedImage) ? '#FF8DAD' : '#CCC'}
              />
            </TouchableOpacity>
          </View>
        </View>

        {/* Image Options Modal */}
        <Modal
          visible={showImageOptions}
          transparent={true}
          animationType="slide"
          onRequestClose={() => setShowImageOptions(false)}
        >
          <TouchableOpacity
            style={styles.modalOverlay}
            activeOpacity={1}
            onPress={() => setShowImageOptions(false)}
          >
            <View style={styles.modalContent}>
              <Text style={styles.modalTitle}>
                {language === 'vi' ? 'Chọn ảnh để phân tích lừa đảo' : 'Select image for fraud analysis'}
              </Text>

              <TouchableOpacity style={styles.modalOption} onPress={takePhoto}>
                <Ionicons name="camera" size={28} color="#FF8DAD" />
                <Text style={styles.modalOptionText}>
                  {language === 'vi' ? 'Chụp ảnh' : 'Take Photo'}
                </Text>
              </TouchableOpacity>

              <TouchableOpacity style={styles.modalOption} onPress={pickImage}>
                <Ionicons name="images" size={28} color="#FF8DAD" />
                <Text style={styles.modalOptionText}>
                  {language === 'vi' ? 'Chọn từ thư viện' : 'Choose from Library'}
                </Text>
              </TouchableOpacity>

              <TouchableOpacity
                style={styles.modalCancel}
                onPress={() => setShowImageOptions(false)}
              >
                <Text style={styles.modalCancelText}>
                  {language === 'vi' ? 'Hủy' : 'Cancel'}
                </Text>
              </TouchableOpacity>
            </View>
          </TouchableOpacity>
        </Modal>
      </KeyboardAvoidingView>
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
  imagePreviewContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(255, 255, 255, 0.9)',
    marginHorizontal: 15,
    marginBottom: 5,
    borderRadius: 15,
    padding: 10,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 3,
    elevation: 3
  },
  imagePreview: {
    width: 60,
    height: 60,
    borderRadius: 10
  },
  imagePreviewInfo: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginLeft: 15
  },
  imagePreviewText: {
    color: '#666',
    fontSize: 14
  },
  removeImageBtn: {
    padding: 5
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    justifyContent: 'flex-end'
  },
  modalContent: {
    backgroundColor: '#FFF',
    borderTopLeftRadius: 25,
    borderTopRightRadius: 25,
    padding: 25,
    paddingBottom: 40
  },
  modalTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#333',
    textAlign: 'center',
    marginBottom: 25
  },
  modalOption: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 15,
    paddingHorizontal: 20,
    backgroundColor: '#FFF5F8',
    borderRadius: 15,
    marginBottom: 12
  },
  modalOptionText: {
    fontSize: 16,
    color: '#333',
    marginLeft: 15
  },
  modalCancel: {
    alignItems: 'center',
    paddingVertical: 15,
    marginTop: 10
  },
  modalCancelText: {
    fontSize: 16,
    color: '#999'
  }
});
