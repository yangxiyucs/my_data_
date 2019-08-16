# 2.3.初识Appium(简述工作原理)

* 2.3.1 Android端

```
	1. Android端，appium基于WebDriver，在设备中注⼊Bootstrap.jar，
		通过调用UiAutomator的命令，实现App自动化测试;
	2. UiAutomator测试框架是Android SDK⾃自带的App UI自动化测试Java库
	3. 由于UiAutomator对H5的⽀支持有限，appium引入了chromedriver来实现基于H5的自动化
```
![android_appium](../m_image/android_appium.png)

* 2.3.2 IOS端

```
	1.IOS端，appium同样基于WebDriver,appium ios封装了apple Instruments框架，
		在设备中注⼊bootstrap.js，实现App自动化测试
	2.使用Instruments框架里的UI Automation(Apple的⾃自动化测试框架)
```
![ios_appium](../m_image/ios_appium.png)