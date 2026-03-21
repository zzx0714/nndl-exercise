

import numpy as np
import matplotlib.pyplot as plt


def target_function(x):
    return 0.5 * x**2 + 0.5 * x + 1



def generate_dataset(x_range=(-5, 5), num_samples=100, seed=42):
    
    np.random.seed(seed)
    
    # 生成均匀分布的x值
    x = np.random.uniform(x_range[0], x_range[1], num_samples)
    y = target_function(x)
    
    # 添加小的噪声
    noise = np.random.normal(0, 0.05, num_samples)
    y = y + noise
    
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    
    # 数据归一化
    x_min, x_max = x_range
    x = (x - x_min) / (x_max - x_min) * 2 - 1  # 归一化到 [-1, 1]
    
    return x, y



class ReLUNetwork:
    
    def __init__(self, input_size=1, hidden_size=50, output_size=1, learning_rate=0.01):
        
        self.learning_rate = learning_rate
        
        
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        
        
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
        
        self.loss_history = []
    
    def relu(self, x):
        """ReLU激活函数"""
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """ReLU导数"""
        return (x > 0).astype(float)
    
    def forward(self, X):
        """前向传播"""
        # 隐藏层
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        
        # 输出层
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.y_pred = self.z2  # 线性激活
        
        return self.y_pred
    
    def backward(self, X, y, batch_size):
        """反向传播 梯度裁剪"""
        m = X.shape[0]
        
        # 输出层梯度
        dz2 = self.y_pred - y  # 均方差损失函数的导数
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        # 隐藏层梯度
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.relu_derivative(self.z1)
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        # 梯度裁剪
        clip_value = 1.0
        dW1 = np.clip(dW1, -clip_value, clip_value)
        db1 = np.clip(db1, -clip_value, clip_value)
        dW2 = np.clip(dW2, -clip_value, clip_value)
        db2 = np.clip(db2, -clip_value, clip_value)
        
        # 更新参数
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
    
    def compute_loss(self, y_true, y_pred):
        """计算均方误差"""
        return np.mean((y_true - y_pred) ** 2) / 2
    
    def train(self, X_train, y_train, epochs=1000, batch_size=32):
        """
        训练网络
        """
        num_samples = X_train.shape[0]
        
        for epoch in range(epochs):
            # 随机打乱数据
            indices = np.random.permutation(num_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            # 小批量梯度下降
            for i in range(0, num_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                self.forward(X_batch)
                self.backward(X_batch, y_batch, batch_size)
            
            # 计算整个数据集的损失
            y_pred = self.forward(X_train)
            loss = self.compute_loss(y_train, y_pred)
            self.loss_history.append(loss)
            
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.6f}")
    
    def predict(self, X):
        """预测"""
        return self.forward(X)
    
    def evaluate(self, X_test, y_test):
        """评估模型"""
        y_pred = self.predict(X_test)
        mse = np.mean((y_test - y_pred) ** 2)
        rmse = np.sqrt(mse)
        return mse, rmse, y_pred


def main():
    print("=" * 60)
    print("ReLU神经网络函数拟合")
    print("=" * 60)
    print("目标函数: y = 0.5*x^2 + 0.5*x + 1")
    print("=" * 60)
    
    # 生成数据集
    print("\n1. 生成数据集...")
    X_train, y_train = generate_dataset(x_range=(-5, 5), num_samples=200, seed=42)
    X_test, y_test = generate_dataset(x_range=(-5, 5), num_samples=100, seed=123)
    
    print(f"   训练集大小: {X_train.shape[0]}")
    print(f"   测试集大小: {X_test.shape[0]}")
    
    # 保存数据集到文件
    print("\n2. 保存数据集...")
    # 逆归一化后保存
    X_train_save = X_train * 5 - 1
    X_test_save = X_test * 5 - 1
    np.savetxt('train.txt', np.hstack([X_train_save, y_train]), 
               fmt='%.6f', header='x y', comments='')
    np.savetxt('test.txt', np.hstack([X_test_save, y_test]), 
               fmt='%.6f', header='x y', comments='')
    print("   训练集已保存到 train.txt")
    print("   测试集已保存到 test.txt")
    
    # 创建并训练网络
    print("\n3. 创建神经网络...")
    model = ReLUNetwork(input_size=1, hidden_size=50, output_size=1, learning_rate=0.01)
    print("   网络结构: 输入(1) -> 隐藏层(50, ReLU) -> 输出(1, 线性)")
    
    print("\n4. 训练网络...")
    model.train(X_train, y_train, epochs=1000, batch_size=16)
    
    # 评估模型
    print("\n5. 评估模型...")
    train_mse, train_rmse, _ = model.evaluate(X_train, y_train)
    test_mse, test_rmse, y_pred_test = model.evaluate(X_test, y_test)
    
    print(f"\n   训练集 MSE: {train_mse:.6f}, RMSE: {train_rmse:.6f}")
    print(f"   测试集 MSE: {test_mse:.6f}, RMSE: {test_rmse:.6f}")
    
  
    print("\n6. 绘制结果...")
    plt.figure(figsize=(14, 5))
    
 
    X_train_orig = X_train * 5 - 1  
    X_test_orig = X_test * 5 - 1
    
    # 图1：损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(model.loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training Loss Curve')
    plt.grid(True)
    
    # 图2：训练集拟合效果
    plt.subplot(1, 2, 2)
    y_pred_train = model.predict(X_train)
    plt.scatter(X_train_orig, y_train, alpha=0.5, label='Training Data', s=20)
    sorted_indices = np.argsort(X_train_orig.flatten())
    plt.plot(X_train_orig[sorted_indices], y_pred_train[sorted_indices], 
            'r-', linewidth=2, label='Model Prediction')
    x_smooth = np.linspace(-5, 5, 1000).reshape(-1, 1)
    y_smooth = target_function(x_smooth)
    plt.plot(x_smooth, y_smooth, 'g--', linewidth=2, label='Target Function')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Training Set Fitting')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('fitting_result.png', dpi=100)
    print("   结果图已保存到 fitting_result.png")
    plt.show()
    
    print("\n" + "=" * 60)
    print("拟合完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()
