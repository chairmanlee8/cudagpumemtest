#include "gpudisplaywidget.h"
#include "qtgpumemtest.h"

GpuDisplayWidget::GpuDisplayWidget(QWidget *parent)
	: QWidget(parent), widgetIndex(0), state(SelectingMode)
{
	layout = new QVBoxLayout(this);
	innerTopLayout = new QHBoxLayout();
	innerBottomLayout = new QHBoxLayout();

	startButton = new QToolButton();
	startLoopButton = new QToolButton();
	stopButton = new QToolButton();
	resultsButton = new QToolButton();

	startButton->setIcon(QIcon(":/QtGpuMemtest/resources/play_18x24_blue.png"));
	startButton->setIconSize(QSize(24, 24));
	startLoopButton->setIcon(QIcon(":/QtGpuMemtest/resources/playinfinite_18x24_blue.png"));
	startLoopButton->setIconSize(QSize(24, 24));
	stopButton->setIcon(QIcon(":/QtGpuMemtest/resources/stop_16x16_blue.png"));
	stopButton->setIconSize(QSize(24, 24));
	stopButton->setEnabled(false);
	resultsButton->setIcon(QIcon(":/QtGpuMemtest/resources/book_24x24.png"));
	resultsButton->setIconSize(QSize(24, 24));

	labelGpu = new QLabel("Default GPU");
	labelMemory = new QLabel("#Memory");
	progress = new QHBoxLayout();
	progress->setAlignment(Qt::AlignLeft);
	checkStart = new QCheckBox();
	labelStopping = new QLabel();

	progress->setSpacing(1);
	labelGpu->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);

	innerTopLayout->addWidget(checkStart);
	innerTopLayout->addWidget(labelGpu);
	layout->addLayout(innerTopLayout);
	layout->addWidget(labelMemory);
	layout->addLayout(progress);
	innerBottomLayout->addWidget(startButton);
	innerBottomLayout->addWidget(startLoopButton);
	innerBottomLayout->addWidget(stopButton);
	innerBottomLayout->addSpacing(10);
	innerBottomLayout->addWidget(labelStopping);
	innerBottomLayout->addStretch();
	innerBottomLayout->addWidget(resultsButton, 0);
	layout->addLayout(innerBottomLayout);

	this->setStyleSheet
	(
		"GpuDisplayWidget { \
		background-color: qradialgradient(cx: 0, cy: 0, fx: 1, fy: 1, radius: 1, stop: 0 #ffffff, stop: 1 #8BDEFF); \
			margin-top: 1px; \
			margin-bottom: 1px; \
			border: 2px solid #333333; \
			border-radius: 4px \
		} \
		QToolButton { \
			border: none; \
		}"
	);

	// Connect signals and slots
	QSignalMapper* mapper = new QSignalMapper(this);
	mapper->setMapping(startButton, 0);
	mapper->setMapping(startLoopButton, 1);
	connect(startButton, SIGNAL(clicked()), mapper, SLOT(map()));
	connect(startLoopButton, SIGNAL(clicked()), mapper, SLOT(map()));
	connect(mapper, SIGNAL(mapped(int)), this, SIGNAL(startTests(int)));

	connect(stopButton, SIGNAL(clicked()), this, SLOT(stopButtonClicked()));
	connect(resultsButton, SIGNAL(clicked()), this, SIGNAL(displayResults()));

	// Set initial state
	setState(SelectingMode);
}

GpuDisplayWidget::~GpuDisplayWidget()
{

}

/*
void GpuDisplayWidget::startTest(bool infinite)
{
	// Blah blah load test options and what not, configuration and stuff, anything that needs to be
	// passed to the parent window/process.
	emit testStarted(m_widgetIndex, infinite);

	startButton->setEnabled(false);
	startLoopButton->setEnabled(false);
	stopButton->setEnabled(true);

	// Setup GPU thread
	gpuThread = new QtGpuThread(tests);

	connect(gpuThread, SIGNAL(failed(int, QString)), this, SLOT(testFailed(int, QString)));
	connect(gpuThread, SIGNAL(passed(int, QString)), this, SLOT(testPassed(int, QString)));
	connect(gpuThread, SIGNAL(starting(int, QString)), this, SLOT(testStarting(int, QString)));
	connect(gpuThread, SIGNAL(finished()), this, SLOT(endTest()));
	connect(gpuThread, SIGNAL(log(int, QString, QString)), this, SLOT(testLog(int, QString, QString)));

	// Link the controller to test ended signals
	// Notify the controller of how many tests will be performed for this widget
	int enabledCount = 0;
	for(int i = 0; i < tests.count(); i++)
	{
		if(tests[i]->testEnabled) enabledCount++;
	}
	m_controller->testsStarted(enabledCount);
	connect(gpuThread, SIGNAL(ended(int, QString)), m_controller, SLOT(testEnded(int, QString)));

	gpuThread->setDevice(m_widgetIndex);
	if(infinite) gpuThread->setEndless(true);
	gpuThread->start();
}

void GpuDisplayWidget::endTest()
{
	static unsigned int timesEnded = 0;

	if(!stopButton->isEnabled()) return;	// Nothing to stop

	if(timesEnded == 0)
	{
		// Is the thread already stopped? (naturally stopped)
		// then call endTest again
		if(!gpuThread->isRunning())
		{
			timesEnded += 2;
			m_labelStopping->setText(QString("Done, click stop to reset."));
		}
		else
		{
			// Kill thread
			gpuThread->notifyExit();
			m_labelStopping->setText(QString("Stopping..."));
			timesEnded++;
		}
	}
	else if(timesEnded == 1)
	{
		// Do nothing if gpuThread is still running, otherwise increment timesEnded
		if(!gpuThread->isRunning())
		{
			emit testEnded(m_widgetIndex);

			m_labelStopping->setText(QString("Done, click stop again to reset."));
			timesEnded++;
		}
	}
	else if(timesEnded > 1)
	{
		m_labelStopping->setText(QString(""));

		for(int i = 0; i < testWidgets.size(); i++)
		{
			testWidgets[i]->setMode(TestIconWidget::SelectMode);
			testWidgets[i]->setStatus(TestNotStarted);
		}

		startButton->setEnabled(true);
		startLoopButton->setEnabled(true);
		stopButton->setEnabled(false);
		timesEnded = 0;
	}
}

void GpuDisplayWidget::displayLog()
{
	QString wtitle;
	QTextStream t(&wtitle);

	t << "Results [" << m_labelGpu->text() << "]";

	ResultsDisplay *rd = new ResultsDisplay();
	rd->setResults(log);
	rd->setWindowTitle(wtitle);
	rd->show();
}

QString GpuDisplayWidget::getLog()
{
	QString t;
	QTextStream tt(&t);

	for(int i = 0; i < log.size(); i++)
	{
		tt << log[i] << "\r\n";	// TODO: UNIX or CR/LF line endings
	}

	return t;
}*/

void GpuDisplayWidget::setState(Mode newState)
{
	// The current state controls what buttons on the bottom are available to the user.
	// Additionally, it controls the display mode of the TestIconWidgets.

	switch(newState)
	{
		case SelectingMode:
			startButton->setEnabled(true);
			startLoopButton->setEnabled(true);
			stopButton->setEnabled(false);
			labelStopping->setText(tr(""));
			break;
		case RunningMode:
			startButton->setEnabled(false);
			startLoopButton->setEnabled(false);
			stopButton->setEnabled(true);
			labelStopping->setText(tr(""));
			break;
		case StoppedMode:
			startButton->setEnabled(false);
			startLoopButton->setEnabled(false);
			stopButton->setEnabled(true);
			labelStopping->setText(tr("Done, press stop again to reset tests."));
			break;
	}

	for(int i = 0; i < testWidgets.count(); i++)
	{
		testWidgets[i]->setState((newState == SelectingMode) ? TestIconWidget::SelectMode : TestIconWidget::DisplayMode);
	}

	state = newState;
}

void GpuDisplayWidget::testFailed(TestInfo test)
{
	for(int i = 0; i < testWidgets.size(); i++)
	{
		if(test == testWidgets[i]->getTestInfo())
		{
			testWidgets[i]->setStatus(TestFailed);
			break;
		}
	}
}

void GpuDisplayWidget::testPassed(TestInfo test)
{
	for(int i = 0; i < testWidgets.size(); i++)
	{
		if(test == testWidgets[i]->getTestInfo())
		{
			testWidgets[i]->setStatus(TestPassed);
			break;
		}
	}
}

void GpuDisplayWidget::testStarting(TestInfo test)
{
	for(int i = 0; i < testWidgets.size(); i++)
	{
		if(test == testWidgets[i]->getTestInfo())
		{
			testWidgets[i]->setStatus(TestRunning);
			break;
		}
	}
}

void GpuDisplayWidget::setFont(const QFont &font)
{
	labelGpu->setFont(font);

	QFont *newFont = new QFont(font);
	newFont->setPointSize(newFont->pointSize() - 4);

	labelMemory->setFont(*newFont);
}

void GpuDisplayWidget::setTests(QVector<TestInfo>& aTests)
{
	// Remove old tests and add new ones
	for(int i = 0; i < testWidgets.size(); i++)
	{
		progress->removeWidget(testWidgets[i]);
		delete testWidgets[i];
	}

	testWidgets.clear();

	for(int i = 0; i < aTests.size(); i++)
	{
		TestIconWidget* aWidget = new TestIconWidget(aTests[i]);
		connect(aWidget, SIGNAL(maskEnable(TestInfo, bool)), this, SLOT(maskSelect(TestInfo, bool)));
		//aWidget->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);

		progress->addWidget(aWidget);
		testWidgets.push_back(aWidget);
	}
}

QVector<TestInfo> GpuDisplayWidget::getTests()
{
	QVector<TestInfo> temp;
	for(int i = 0; i < testWidgets.size(); i++)
		temp.append(testWidgets[i]->getTestInfo());
	return temp;
}

bool GpuDisplayWidget::isTestFailed()
{
	for(int i = 0; i < testWidgets.size(); i++)
	{
		if(testWidgets[i]->getStatus() == TestFailed)
			return true;
	}

	return false;
}

void GpuDisplayWidget::stopButtonClicked()
{
	switch(state)
	{
		case RunningMode:
			emit stopTests();
			break;
		case StoppedMode:
			setState(SelectingMode);
			break;
		case SelectingMode:
			break;
	}
}

void GpuDisplayWidget::maskSelect(TestInfo pivot, bool mask)
{
	int num_selected = 0;

	// If double click is on a solo-ed item, select all
	// If double click is on a non-solo selected item, solo it
	// If double click is on a non-selected item, select all except it (is this intuitive?)

	for(int i = 0; i < testWidgets.size(); i++)
	{
		if(testWidgets[i]->getTestInfo().testEnabled) num_selected++;
	}

	for(int i = 0; i < testWidgets.size(); i++)
	{
		if(testWidgets[i]->getTestInfo() != pivot)
		{
			testWidgets[i]->setTestEnabled(((num_selected == 1) ^ !mask) | !mask);
		}
	}

	update();
}

void GpuDisplayWidget::paintEvent(QPaintEvent* event)
{
	QStyleOption opt;
	opt.init(this);
	QPainter p(this);
	style()->drawPrimitive(QStyle::PE_Widget, &opt, &p, this);
	//QWidget::paintEvent(event);
}