/****************************************************************************
** Meta object code from reading C++ file 'testiconwidget.h'
**
** Created: Tue May 3 15:01:32 2011
**      by: The Qt Meta Object Compiler version 62 (Qt 4.7.1)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../testiconwidget.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'testiconwidget.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 62
#error "This file was generated using the moc from 4.7.1. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_TestIconWidget[] = {

 // content:
       5,       // revision
       0,       // classname
       0,    0, // classinfo
       8,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: signature, parameters, type, tag, flags
      24,   16,   15,   15, 0x0a,
      57,   15,   46,   15, 0x0a,
      78,   15,   69,   15, 0x0a,
      94,   92,   15,   15, 0x0a,
     113,   15,  108,   15, 0x0a,
     125,  123,   15,   15, 0x0a,
     143,   15,  139,   15, 0x0a,
     154,   15,   15,   15, 0x0a,

       0        // eod
};

static const char qt_meta_stringdata_TestIconWidget[] = {
    "TestIconWidget\0\0aStatus\0setStatus(TestStatus)\0"
    "TestStatus\0getStatus()\0TestInfo\0"
    "getTestInfo()\0m\0setMode(Mode)\0Mode\0"
    "getMode()\0a\0setWidth(int)\0int\0getWidth()\0"
    "updateRunningColor()\0"
};

const QMetaObject TestIconWidget::staticMetaObject = {
    { &QWidget::staticMetaObject, qt_meta_stringdata_TestIconWidget,
      qt_meta_data_TestIconWidget, 0 }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &TestIconWidget::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *TestIconWidget::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *TestIconWidget::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_TestIconWidget))
        return static_cast<void*>(const_cast< TestIconWidget*>(this));
    return QWidget::qt_metacast(_clname);
}

int TestIconWidget::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QWidget::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        switch (_id) {
        case 0: setStatus((*reinterpret_cast< TestStatus(*)>(_a[1]))); break;
        case 1: { TestStatus _r = getStatus();
            if (_a[0]) *reinterpret_cast< TestStatus*>(_a[0]) = _r; }  break;
        case 2: { TestInfo _r = getTestInfo();
            if (_a[0]) *reinterpret_cast< TestInfo*>(_a[0]) = _r; }  break;
        case 3: setMode((*reinterpret_cast< Mode(*)>(_a[1]))); break;
        case 4: { Mode _r = getMode();
            if (_a[0]) *reinterpret_cast< Mode*>(_a[0]) = _r; }  break;
        case 5: setWidth((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 6: { int _r = getWidth();
            if (_a[0]) *reinterpret_cast< int*>(_a[0]) = _r; }  break;
        case 7: updateRunningColor(); break;
        default: ;
        }
        _id -= 8;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
