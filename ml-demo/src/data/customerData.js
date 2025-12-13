/**
 * Dữ liệu 3 khách hàng demo
 * =============================
 * File này chứa dữ liệu của 3 khách hàng mẫu để sử dụng trong phần "Kiểm tra giao dịch"
 * Các user ID: USR_0000001, USR_0000002, USR_0000003
 */

// =============== KHÁCH HÀNG 1: USR_0000001 ===============
export const USER_0000001 = {
  user_id: 'USR_0000001',
  name: 'Nguyễn Văn An',
  age: 32,
  gender: 'M',
  occupation: 'Kỹ sư phần mềm',
  income_level: 'high',
  account_age_days: 1095,
  city: 'Hà Nội',
  region: 'North',
  phone_verified: true,
  email_verified: true,
  kyc_level: 3,
  avg_monthly_transactions: 45,
  avg_transaction_amount: 8500000,
  preferred_channel: 'mobile',
  device_count: 2,
  login_frequency: 'daily',
  last_login_days_ago: 0,
  historical_risk_score: 0.08,
  is_premium: true,
  created_at: '2021-12-13',
  total_transactions: 540,
  // Behavioral features mặc định
  behavioral_features: {
    velocity_1h: 2,
    velocity_24h: 5,
    time_since_last_transaction: 2.5,
    amount_deviation_ratio: 1.1
  }
};

// Lịch sử giao dịch của USR_0000001
export const TRANSACTIONS_0000001 = [
  {
    transaction_id: 'TXN_0000001_001',
    timestamp: '2024-12-13T09:15:00',
    amount: 5500000,
    transaction_type: 'transfer',
    recipient_id: 'USR_0000002',
    channel: 'mobile_app',
    status: 'completed',
    device_type: 'android',
    location: 'Hanoi'
  },
  {
    transaction_id: 'TXN_0000001_002',
    timestamp: '2024-12-12T14:30:00',
    amount: 1200000,
    transaction_type: 'payment',
    recipient_id: 'MER_SHOPEE',
    channel: 'mobile_app',
    status: 'completed',
    device_type: 'android',
    location: 'Hanoi'
  },
  {
    transaction_id: 'TXN_0000001_003',
    timestamp: '2024-12-11T10:00:00',
    amount: 3500000,
    transaction_type: 'transfer',
    recipient_id: 'USR_0000003',
    channel: 'web_banking',
    status: 'completed',
    device_type: 'windows',
    location: 'Hanoi'
  },
  {
    transaction_id: 'TXN_0000001_004',
    timestamp: '2024-12-10T16:45:00',
    amount: 850000,
    transaction_type: 'payment',
    recipient_id: 'MER_GRAB',
    channel: 'mobile_app',
    status: 'completed',
    device_type: 'android',
    location: 'Hanoi'
  },
  {
    transaction_id: 'TXN_0000001_005',
    timestamp: '2024-12-09T08:20:00',
    amount: 15000000,
    transaction_type: 'transfer',
    recipient_id: 'USR_EXTERNAL',
    channel: 'web_banking',
    status: 'completed',
    device_type: 'windows',
    location: 'Hanoi'
  }
];

// =============== KHÁCH HÀNG 2: USR_0000002 ===============
export const USER_0000002 = {
  user_id: 'USR_0000002',
  name: 'Trần Thị Bình',
  age: 28,
  gender: 'F',
  occupation: 'Nhân viên ngân hàng',
  income_level: 'medium',
  account_age_days: 730,
  city: 'Hồ Chí Minh',
  region: 'South',
  phone_verified: true,
  email_verified: true,
  kyc_level: 3,
  avg_monthly_transactions: 35,
  avg_transaction_amount: 4500000,
  preferred_channel: 'mobile',
  device_count: 1,
  login_frequency: 'daily',
  last_login_days_ago: 0,
  historical_risk_score: 0.05,
  is_premium: false,
  created_at: '2022-12-13',
  total_transactions: 420,
  // Behavioral features mặc định
  behavioral_features: {
    velocity_1h: 1,
    velocity_24h: 3,
    time_since_last_transaction: 4.0,
    amount_deviation_ratio: 0.9
  }
};

// Lịch sử giao dịch của USR_0000002
export const TRANSACTIONS_0000002 = [
  {
    transaction_id: 'TXN_0000002_001',
    timestamp: '2024-12-13T11:20:00',
    amount: 2800000,
    transaction_type: 'transfer',
    recipient_id: 'USR_0000001',
    channel: 'mobile_app',
    status: 'completed',
    device_type: 'ios',
    location: 'HCMC'
  },
  {
    transaction_id: 'TXN_0000002_002',
    timestamp: '2024-12-12T09:45:00',
    amount: 650000,
    transaction_type: 'payment',
    recipient_id: 'MER_LAZADA',
    channel: 'mobile_app',
    status: 'completed',
    device_type: 'ios',
    location: 'HCMC'
  },
  {
    transaction_id: 'TXN_0000002_003',
    timestamp: '2024-12-11T15:30:00',
    amount: 4200000,
    transaction_type: 'transfer',
    recipient_id: 'USR_FAMILY',
    channel: 'mobile_app',
    status: 'completed',
    device_type: 'ios',
    location: 'HCMC'
  },
  {
    transaction_id: 'TXN_0000002_004',
    timestamp: '2024-12-10T12:00:00',
    amount: 1500000,
    transaction_type: 'payment',
    recipient_id: 'MER_TIKI',
    channel: 'web_banking',
    status: 'completed',
    device_type: 'macos',
    location: 'HCMC'
  },
  {
    transaction_id: 'TXN_0000002_005',
    timestamp: '2024-12-09T18:15:00',
    amount: 8500000,
    transaction_type: 'transfer',
    recipient_id: 'USR_0000003',
    channel: 'mobile_app',
    status: 'completed',
    device_type: 'ios',
    location: 'HCMC'
  }
];

// =============== KHÁCH HÀNG 3: USR_0000003 ===============
export const USER_0000003 = {
  user_id: 'USR_0000003',
  name: 'Lê Hoàng Cường',
  age: 45,
  gender: 'M',
  occupation: 'Giám đốc doanh nghiệp',
  income_level: 'very_high',
  account_age_days: 2190,
  city: 'Đà Nẵng',
  region: 'Central',
  phone_verified: true,
  email_verified: true,
  kyc_level: 3,
  avg_monthly_transactions: 65,
  avg_transaction_amount: 25000000,
  preferred_channel: 'web',
  device_count: 3,
  login_frequency: 'daily',
  last_login_days_ago: 0,
  historical_risk_score: 0.12,
  is_premium: true,
  created_at: '2018-12-13',
  total_transactions: 1560,
  // Behavioral features mặc định
  behavioral_features: {
    velocity_1h: 3,
    velocity_24h: 8,
    time_since_last_transaction: 1.5,
    amount_deviation_ratio: 1.3
  }
};

// Lịch sử giao dịch của USR_0000003
export const TRANSACTIONS_0000003 = [
  {
    transaction_id: 'TXN_0000003_001',
    timestamp: '2024-12-13T08:30:00',
    amount: 35000000,
    transaction_type: 'transfer',
    recipient_id: 'COMPANY_SUPPLIER',
    channel: 'web_banking',
    status: 'completed',
    device_type: 'windows',
    location: 'Danang'
  },
  {
    transaction_id: 'TXN_0000003_002',
    timestamp: '2024-12-12T16:00:00',
    amount: 12000000,
    transaction_type: 'transfer',
    recipient_id: 'USR_0000001',
    channel: 'mobile_app',
    status: 'completed',
    device_type: 'ios',
    location: 'Danang'
  },
  {
    transaction_id: 'TXN_0000003_003',
    timestamp: '2024-12-11T11:45:00',
    amount: 5500000,
    transaction_type: 'payment',
    recipient_id: 'MER_AIRLINE',
    channel: 'web_banking',
    status: 'completed',
    device_type: 'windows',
    location: 'Danang'
  },
  {
    transaction_id: 'TXN_0000003_004',
    timestamp: '2024-12-10T09:15:00',
    amount: 85000000,
    transaction_type: 'transfer',
    recipient_id: 'COMPANY_PARTNER',
    channel: 'web_banking',
    status: 'completed',
    device_type: 'windows',
    location: 'Danang'
  },
  {
    transaction_id: 'TXN_0000003_005',
    timestamp: '2024-12-09T14:30:00',
    amount: 2200000,
    transaction_type: 'payment',
    recipient_id: 'MER_HOTEL',
    channel: 'mobile_app',
    status: 'completed',
    device_type: 'ios',
    location: 'Hanoi'
  }
];

// =============== EXPORT TỔNG HỢP ===============

/**
 * Danh sách 3 khách hàng demo
 */
export const DEMO_CUSTOMERS = [
  USER_0000001,
  USER_0000002,
  USER_0000003
];

/**
 * Map lịch sử giao dịch theo user_id
 */
export const DEMO_TRANSACTIONS = {
  'USR_0000001': TRANSACTIONS_0000001,
  'USR_0000002': TRANSACTIONS_0000002,
  'USR_0000003': TRANSACTIONS_0000003
};

/**
 * Lấy thông tin user theo ID
 */
export const getDemoUserById = (userId) => {
  return DEMO_CUSTOMERS.find(user => user.user_id === userId) || null;
};

/**
 * Lấy lịch sử giao dịch theo user ID
 */
export const getDemoTransactions = (userId) => {
  return DEMO_TRANSACTIONS[userId] || [];
};

/**
 * Lấy profile user (format giống API response)
 */
export const getDemoUserProfile = (userId) => {
  const user = getDemoUserById(userId);
  if (!user) return null;

  return {
    name: user.name,
    age: user.age,
    occupation: user.occupation,
    income_level: user.income_level,
    account_age_days: user.account_age_days,
    kyc_level: user.kyc_level,
    avg_transaction_amount: user.avg_transaction_amount,
    historical_risk_score: user.historical_risk_score,
    total_transactions: user.total_transactions,
    city: user.city,
    is_premium: user.is_premium
  };
};

/**
 * Lấy behavioral features theo user ID
 */
export const getDemoBehavioralFeatures = (userId) => {
  const user = getDemoUserById(userId);
  if (!user) return null;
  return user.behavioral_features;
};

export default {
  DEMO_CUSTOMERS,
  DEMO_TRANSACTIONS,
  getDemoUserById,
  getDemoTransactions,
  getDemoUserProfile,
  getDemoBehavioralFeatures
};
