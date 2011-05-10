/****************************************************************************
** Meta object code from reading C++ file 'gpudisplaywidget.h'
**
** Created: Tue May 3 15:54:55 2011
**      by: The Qt Meta Object Compiler version 62 (Qt 4.7.1)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../../../GpuDisplayWidget/GpuDisplayWidget/gpudisplaywidget.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'gpudisplaywidget.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 62
#error "This file was generated using the moc from 4.7.1. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_GpuDisplayWidget[] = {

 // content:
       5,       // revision
       0,       // classname
       0,    0, // classinfo
      19,   14, // methods
       1,  109, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       2,       // signalCount

 // signals: signature, parameters, type, tag, flags
      33,   18,   17,   17, 0x05,
      61,   55,   17,   17, 0x05,

 // slots: signature, parameters, type, tag, flags
      84,   76,   17,   17, 0x0a,
     114,  104,   17,   17, 0x0a,
     150,  136,   17,   17, 0x0a,
     184,  176,   17,   17, 0x0a,
     210,  203,   17,   17, 0x0a,
     248,  239,   17,   17, 0x0a,
     264,   17,   17,   17, 0x2a,
     276,   17,   17,   17, 0x0a,
     292,   17,   17,   17, 0x0a,
     312,   17,   17,   17, 0x0a,
     322,   17,   17,   17, 0x0a,
     354,  335,   17,   17, 0x0a,
     378,  335,   17,   17, 0x0a,
     402,  335,   17,   17, 0x0a,
     458,  428,   17,   17, 0x0a,
     492,   17,  487,   17, 0x0a,
     504,   17,  487,   17, 0x0a,

 // properties: name, type, flags
     525,  519, 0x40095103,

       0        // eod
};

static const char qt_meta_stringdata_GpuDisplayWidget[] = {
    "GpuDisplayWidget\0\0index,infinite\0"
    "testStarted(int,bool)\0index\0testEnded(int)\0"
    "gpuName\0setGpuName(QString)\0gpuMemory\0"
    "setGpuMemory(QString)\0gpuTestStatus\0"
    "setTestStatus(TestStatus)\0checked\0"
    "setCheckStart(int)\0aTests\0"
    "setTests(QVector<TestInfo>&)\0infinite\0"
    "startTest(bool)\0startTest()\0startTestOnce()\0"
    "startTestInfinite()\0endTest()\0"
    "displayLog()\0deviceIdx,testName\0"
    "testFailed(int,QString)\0testPassed(int,QString)\0"
    "testStarting(int,QString)\0"
    "deviceIdx,testName,logMessage\0"
    "testLog(int,QString,QString)\0bool\0"
    "isChecked()\0isTestFailed()\0QFont\0font\0"
};

const QMetaObject GpuDisplayWidget::staticMetaObject = {
    { &QWidget::staticMetaObject, qt_meta_stringdata_GpuDisplayWidget,
      qt_meta_data_GpuDisplayWidget, 0 }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &GpuDisplayWidget::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *GpuDisplayWidget::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *GpuDisplayWidget::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_GpuDisplayWidget))
        return static_cast<void*>(const_cast< GpuDisplayWidget*>(this));
    return QWidget::qt_metacast(_clname);
}

int GpuDisplayWidget::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        switch (_id) {
        case 0: testStarted((*reinterpret_cast< const int(*)>(_a[1])),(*reinterpret_cast< bool(*)>(_a[2]))); break;
        case 1: testEnded((*reinterpret_cast< const int(*)>(_a[1]))); break;
        case 2: setGpuName((*reinterpret_cast< const QString(*)>(_a[1]))); break;
        case 3: setGpuMemory((*reinterpret_cast< const QString(*)>(_a[1]))); break;
        case 4: setTestStatus((*reinterpret_cast< const TestStatus(*)>(_a[1]))); break;
        case 5: setCheckStart((*reinterpret_cast< const int(*)>(_a[1]))); break;
        case 6: setTests((*reinterpret_cast< QVector<TestInfo>(*)>(_a[1]))); break;
        case 7: startTest((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 8: startTest(); break;
        case 9: startTestOnce(); break;
        case 10: startTestInfinite(); break;
        case 11: endTest(); break;
        case 12: displayLog(); break;
        case 13: testFailed((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< QString(*)>(_a[2]))); break;
        case 14: testPassed((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< QString(*)>(_a[2]))); break;
        case 15: testStarting((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< QString(*)>(_a[2]))); break;
        case 16: testLog((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< QString(*)>(_a[2])),(*reinterpret_cast< QString(*)>(_a[3]))); break;
        case 17: { bool _r = isChecked();
            if (_a[0]) *reinterpret_cast< bool*>(_a[0]) = _r; }  break;
        case 18: { bool _r = isTestFailed();
            if (_a[0]) *reinterpret_cast< bool*>(_a[0]) = _r; }  break;
        default: ;
        }
        _id -= 19;
    }
#ifndef QT_NO_PROPERTIES
      else if (_c == QMetaObject::ReadProperty) {
        void *_v = _a[0];
        switch (_id) {
        case 0: *reinterpret_cast< QFont*>(_v) = font(); break;
        }
        _id -= 1;
    } else if (_c == QMetaObject::WriteProperty) {
        void *_v = _a[0];
        switch (_id) {
        case 0: setFont(*reinterpret_cast< QFont*>(_v)); break;
        }
        _id -= 1;
    } else if (_c == QMetaObject::ResetProperty) {
        _id -= 1;
    } else if (_c == QMetaObject::QueryPropertyDesignable) {
        _id -= 1;
    } else if (_c == QMetaObject::QueryPropertyScriptable) {
        _id -= 1;
    } else if (_c == QMetaObject::QueryPropertyStored) {
        _id -= 1;
    } else if (_c == QMetaObject::QueryPropertyEditable) {
        _id -= 1;
    } else if (_c == QMetaObject::QueryPropertyUser) {
        _id -= 1;
    }
#endif // QT_NO_PROPERTIES
    return _id;
}

// SIGNAL 0
void GpuDisplayWidget::testStarted(const int _t1, bool _t2)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)), const_cast<void*>(reinterpret_cast<const void*>(&_t2)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}

// SIGNAL 1
void GpuDisplayWidget::testEnded(const int _t1)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 1, _a);
}
QT_END_MOC_NAMESPACE
