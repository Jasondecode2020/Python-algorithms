import CourseIcon from '@/assets/omni2.0-course-icon.svg';
import UniIcon from '@/assets/omni2.0-uni-icon.svg';
import UniversityModal from '@/components/UniversityModal/UniversityModal';
import {
	UNIVERSITY_LIST
} from '@/constants/storeLocation';
import {
	TIMETABLE_TYPE, COURSE_MESSAGE
} from '@/constants/timetable';
import {
	IUniversityCourseBase
} from '@/interfaces/course';
import {
	ITimetableModelState, ITimetableState
} from '@/interfaces/timetable';
import {
	IUniversityBase, UniversityModelState
} from '@/interfaces/university';
import {
	get as storeGet,
	getSessionStorage,
	has as storeHas,
	removeSessionStorage
} from '@/utils/storageHelper';
import {
	compareOption
} from '@/utils/university';
import {
	Breadcrumb, Button, Card, Tabs, Typography
} from 'antd';
import {
	default as React, FC, useEffect, useState
} from 'react';
import {
	connect, ConnectProps, Link, useIntl
} from 'umi';
import Lecture from './components/Lecture';
import EmptyArrangement from './components/EmptyArrangement';
import ExamForm from './components/ExamForm';
import ExamList from './components/ExamList';
import HomeworkForm from './components/HomeworkForm';
import HomeworkList from './components/HomeworkList';
import CourseList from './components/CourseList';
import CourseModal from './components/Modals/CourseModal';
import styles from './TimetableDetail.less';

export interface ICreateNew {
	homework: boolean,
	course: boolean,
	exam: boolean,
}
interface TimetableDetailProps extends ConnectProps {
	courseListOfMyUniversity: Array<IUniversityCourseBase>,
	timetables: Array<ITimetableState>
}

const TimetableDetail: FC<TimetableDetailProps> = ({ dispatch, courseListOfMyUniversity, timetables }) => {
	const { Text } = Typography;
	const { TabPane } = Tabs;
	const intl = useIntl();
	const { formatMessage } = intl;
	const [showCreateClassTimetable, setShowCreateClassTimetable] = useState(false);
	const [isUniversityModalVisible, setIsUniversityModalVisible] = useState(false);
	const [universities, setUniversities] = useState([]);
	const [savedUniversityId, setSavedUniversityId] = useState(getSessionStorage('selectedUniversity')?._id);
	const [isClickedSaveUniversity, setIsClickedSaveUniversity] = useState(false);
	const [showUniversity, setShowUniversity] = useState('');
	const [showCourse, setShowCourse] = useState('');
	const [isChineseName, setIsChineseName] = useState(true);
	const [coursesOfCurrentUniversity, setCoursesOfCurrentUniversity] = useState([]);
	const [savedCourseId, setSavedCourseId] = useState(getSessionStorage('selectedCourseOfCurrentUniversity')?._id);
	const [isClickedSaveCourse, setIsClickedSaveCourse] = useState(false);
	const [isCourseModalVisible, setIsCourseModalVisible] = useState(false);
	const [currentId, setCurrentId] = useState();
	const [isFromCourseTimetable, setIsFromCourseTimetable] = useState(true);
	const [isEmpty, setIsEmpty] = useState({
		homework: true,
		course: true,
		exam: true
	});
	const [homework, setHomework] = useState<Array<ITimetableState>>([]);
	const [exams, setExams] = useState<Array<ITimetableState>>([]);
	const [courses, setCourses] = useState<Array<ITimetableState>>([]);
	const handleCreateClassTimetableOnClick = () => {
		setShowCreateClassTimetable(true);
	};
	const [createNew, setCreateNew] = useState<ICreateNew>({
		homework: false,
		course: false,
		exam: false
	});
	const [editing, setEditing] = useState<ICreateNew>({
		homework: false,
		course: false,
		exam: false
	});
	const panes = [
		{
			title: formatMessage({
				id: 'timetable.courses-total',
				defaultMessage: 'Course Arrangement'
			}),
			buttonTitle: formatMessage({
				id: 'timetable.detail.button.add.course',
				defaultMessage: 'Add Course'
			}),
			handleButtonOnClick: handleCreateClassTimetableOnClick,
			key: 'course',
			form: Lecture
		},
		{
			title: formatMessage({
				id: 'timetable.assignments-total',
				defaultMessage: 'Homework Arrangement'
			}),
			buttonTitle: formatMessage({
				id: 'timetable.detail.button.add.homework',
				defaultMessage: 'Add Homework'
			}),
			key: 'homework',
			form: HomeworkForm
		},
		{
			title: formatMessage({
				id: 'timetable.exams-total',
				defaultMessage: 'Exam Arrangement'
			}),
			buttonTitle: formatMessage({
				id: 'timetable.detail.button.add.exam',
				defaultMessage: 'Add Exam'
			}),
			key: 'exam',
			form: ExamForm
		}
	];
	useEffect(() => {
		document.getElementsByClassName('ant-tabs-tab-active')[0].parentElement.classList.add('tabs-wrapper');
		return () => {
			removeSessionStorage('selectedCourseOfCurrentUniversity');
			removeSessionStorage('selectedUniversity');
		};
	}, []);

	useEffect(() => {
		if (isFromCourseTimetable) {
			setSavedUniversityId(getSessionStorage('selectedUniversity')?._id);
			setSavedCourseId(getSessionStorage('selectedCourseOfCurrentUniversity')?._id);
		}
	}, [savedUniversityId, savedCourseId]);

	useEffect(() => {
		const universityList: Array<IUniversityBase> = storeHas(UNIVERSITY_LIST) ? storeGet(UNIVERSITY_LIST) : [];
		setUniversities(universityList.sort(compareOption.chineseCompareAscending));
	}, []);

	useEffect(() => {
		setShowCourse((courseListOfMyUniversity[0]?.courseCode || COURSE_MESSAGE.NO_COURSE) + ' ' + (courseListOfMyUniversity[0]?.courseTitle || COURSE_MESSAGE.ADD_COURSE));
		setSavedCourseId(courseListOfMyUniversity[0]?._id)
		setCoursesOfCurrentUniversity(courseListOfMyUniversity);
		setIsClickedSaveCourse(true);
	}, [courseListOfMyUniversity])

	useEffect(()=>{
		dispatch({
			type: 'timetable/fetchTimetables'
		});
	}, [])

	useEffect(()=>{
		const uniId = savedUniversityId;
		const courseId = savedCourseId;
		const myArrangements = timetables.filter(timetable =>
			uniId === timetable.university && timetable.course === courseId
		);

		const filteredHomework = myArrangements.filter(timetable => timetable.type === TIMETABLE_TYPE.ASSIGNMENT
		);
		setHomework(filteredHomework);
		filteredHomework.length ? setIsEmpty((prev)=> ({
			...prev, homework: false
		})) : setIsEmpty((prev)=> ({
			...prev, homework: true
		}));

		const filteredExams = myArrangements.filter(timetable => timetable.type === TIMETABLE_TYPE.EXAM);
		setExams(filteredExams);
		filteredExams.length ? setIsEmpty((prev)=> ({
			...prev, exam: false
		})) : setIsEmpty((prev)=> ({
			...prev, exam: true
		}));

		const filteredCourses = myArrangements.filter(timetable => timetable.type === TIMETABLE_TYPE.LECTURE);
		setCourses(filteredCourses);
		filteredCourses.length ? setIsEmpty((prev) => ({
			...prev, course: false
		})) : setIsEmpty((prev) => ({
			...prev, course: true
		}))

		setIsClickedSaveCourse(false);
		setIsClickedSaveUniversity(false);
	}, [timetables, isClickedSaveCourse, isClickedSaveUniversity])

	// change current university
	const chooseUniversity = (value: string) => {
		setIsFromCourseTimetable(false);
		setSavedUniversityId(value);
	}
	const showChangeUniversityModal = () => setIsUniversityModalVisible(true);
	const handleCancelUniversityModel = () => setIsUniversityModalVisible(false);
	const handleSaveUniversity = () => {
		setIsClickedSaveUniversity(true);
		setIsFromCourseTimetable(false);
		removeSessionStorage('selectedCourseOfCurrentUniversity');
		setIsUniversityModalVisible(false);
		const queenslandUniveristyId = universities.filter(university => university.slug === 'university-of-queensland')[0]?._id;
		dispatch({
			type: 'university/fetchCourseList',
			payload: savedUniversityId || queenslandUniveristyId
		});
		setShowUniversity(isChineseName ?
			universities.filter(university => university._id === savedUniversityId)[0]?.chineseName || 'no value'
			: universities.filter(university => university._id === savedUniversityId)[0]?.name || 'no value');
	};

	// change current course of the university
	const chooseCourse = (value: string) => {
		setIsFromCourseTimetable(false);
		setSavedCourseId(value);
	}
	const showModifyUniversityModal = () => setIsCourseModalVisible(true);
	const handleCancelCourseModel = () => setIsCourseModalVisible(false);
	const handleModifyCourse = () => {
		setIsClickedSaveCourse(true);
		removeSessionStorage('selectedCourseOfCurrentUniversity');
		setIsCourseModalVisible(false);
		if (!savedCourseId) return;
		const showCourseContent = coursesOfCurrentUniversity.filter(course => course._id === savedCourseId)[0];
		setShowCourse(showCourseContent?.courseCode + ' ' + showCourseContent?.courseTitle);
	}
	const edit = (type, id) => {
		setCurrentId(id);
		setCreateNew({
			...createNew, [type]: false
		});
		setEditing({
			...editing, [type]: true
		});
	}

	return (
		<Card
			bordered={false}>
			<UniversityModal visible={isUniversityModalVisible}
				onOk={handleSaveUniversity}
				onCancel={handleCancelUniversityModel}
				onChange={chooseUniversity}
				universities={universities}
			/>
			<CourseModal visible={isCourseModalVisible}
				onOk={handleModifyCourse}
				onCancel={handleCancelCourseModel}
				onChange={chooseCourse}
				courses={coursesOfCurrentUniversity}
				showUniversity={showUniversity}
				isChineseName={isChineseName}
			/>
			<Breadcrumb
				className={styles.standardNav}
				separator=">" >
				<Breadcrumb.Item><Link to='/timetable'>{formatMessage({
					id: 'menu.timetable'
				})}</Link></Breadcrumb.Item>
				<Breadcrumb.Item>
					<Text>{
						formatMessage({
							id: 'timetable.course-details'
						})}
					</Text>
				</Breadcrumb.Item>
			</Breadcrumb>
			<Breadcrumb
				className={styles.standardHeader}
				separator='|'>
				<Breadcrumb.Item>{
					formatMessage({
						id: 'menu.application.training.course.information'
					})
				}
				</Breadcrumb.Item>
				<Breadcrumb.Item>
					<img src={UniIcon}></img>
					<Text>
						{showUniversity || (getSessionStorage('selectedUniversity') ? (isChineseName ?
							`${getSessionStorage('selectedUniversity').chineseName}` :
							`${getSessionStorage('selectedUniversity').name}`) : formatMessage({
							id: 'timetable.detail.select.university',
							defaultMessage: 'No Selected University'
						}))}
					</Text>
					<Button onClick={showChangeUniversityModal}>{
						formatMessage({
							id: 'timetable.detail.button.change',
							defaultMessage: 'Change'
						})}</Button>
				</Breadcrumb.Item>
				<Breadcrumb.Item>
					<img src={CourseIcon}></img>
					<Text>
						{(getSessionStorage('selectedCourseOfCurrentUniversity') ?
							`${getSessionStorage('selectedCourseOfCurrentUniversity')?.courseCode} ${getSessionStorage('selectedCourseOfCurrentUniversity')?.courseTitle}` : showCourse)}
					</Text>
					<Button onClick={showModifyUniversityModal}>{
						formatMessage({
							id: 'timetable.detail.button.modify',
							defaultMessage: 'Modify'
						})}</Button>
				</Breadcrumb.Item>
			</Breadcrumb>
			<Tabs onChange={() => (setCreateNew({
				homework: false,
				course: false,
				exam: false
			}))}>
				{panes.map(pane => (
					<TabPane
						tab={
							<>
								{pane.title}
								<Button
									className={styles['pointer-events']}
									onClick={() => {
										setCreateNew({
											...createNew, [pane.key]: true
										});
										setIsEmpty({
											...isEmpty, [pane.key]: false
										});
										setEditing({
											...editing, [pane.key]: false
										})
										setCurrentId(null);
									} }
								>
									{pane.buttonTitle}
								</Button>
							</>
						}
						key={pane.key}>

						{
							isEmpty[pane.key] && !createNew[pane.key] &&
							<EmptyArrangement
								title={pane.title}
								create={() => {
									setCreateNew({
										...createNew, [pane.key]: true
									});
									setIsEmpty(()=>({
										...isEmpty, [pane.key]: false
									}));
								}}
							/>
						}
						{pane.key === TIMETABLE_TYPE.EXAM && exams && exams
							.map(item =>
								<ExamList key={item._id} item={item}
									edit={()=>{
										setCurrentId(item._id);
										setCreateNew((prev)=>({
											...prev, [pane.key]: true
										}))
									}}
									deleteItem={()=>{
										dispatch({
											type: 'timetable/deleteTimetableById',
											payload: item._id
										})
									}} />
							)}
						{pane.key === TIMETABLE_TYPE.HOMEWORK && homework && homework
							.map(item =>
								<HomeworkList homework={item} key={item._id.toString()} edit={()=>edit(pane.key, item._id)} deleteItem={()=>{
									dispatch({
										type: 'timetable/deleteHomeworkById',
										payload: item._id
									})
								}} />)}
						{pane.key === TIMETABLE_TYPE.COURSE && courses && courses
							.map(item =>
								<CourseList courses={item}
									key={item._id.toString()}
									edit={() => {
										setCurrentId(item._id);
										setCreateNew((prev) => ({
											...prev, [pane.key]: true
										}))
									}}
									deleteItem={()=>{
										dispatch({
											type: 'timetable/deleteTimetableById',
											payload: item._id
										})
									}}
								/>
							)
						}
						{pane.key === TIMETABLE_TYPE.HOMEWORK && editing.homework && !createNew.homework &&
							<HomeworkForm
								dispatch={dispatch}
								isEditing={editing.homework}
								setEditing = {setEditing}
								cancel={() => {
									setEditing({
										...editing, [pane.key]: false
									});
									(homework.length === 0) && setIsEmpty({
										...isEmpty, [pane.key]: true
									});
								}}
								homework={homework.filter(item => item._id === currentId)[0]}
							/>
						}
						{pane.key === TIMETABLE_TYPE.HOMEWORK && createNew.homework && !editing.homework &&
							<HomeworkForm
								dispatch={dispatch}
								setCreateNew={setCreateNew}
								cancel={() => {
									setCreateNew({
										...createNew, [pane.key]: false
									});
								}}
								universityId={savedUniversityId}
								courseId={savedCourseId}
								courseCode={coursesOfCurrentUniversity.find(course => course._id === savedCourseId)?.courseCode}
							/>}
						{pane.key === TIMETABLE_TYPE.COURSE && createNew.course && <Lecture
							key={currentId}
							dispatch={dispatch}
							cancel={() => {
								setCreateNew({
									...createNew, [pane.key]: false
								});
								if(courses.length === 0) {
									setIsEmpty({
										...isEmpty, [pane.key]: true
									});
								}
							}}
							course={courses.filter(item=>item._id ===currentId)[0]}
							universityId={savedUniversityId}
							courseId={savedCourseId}
							courseCode={coursesOfCurrentUniversity.find(course => course._id === savedCourseId)?.courseCode}
						/>}
						{pane.key === TIMETABLE_TYPE.EXAM && createNew.exam &&
						<ExamForm
							key={currentId}
							dispatch={dispatch}
							afterCancel={() => {
								setCreateNew({ ...createNew, [pane.key]: false });
								if(exams.length === 0) {
									setIsEmpty({ ...isEmpty, [pane.key]: true });
								}
							}}
							afterSave={() => {
								setCreateNew({...createNew, [pane.key]: false})
								setIsEmpty({...isEmpty, [pane.key]: false});
							}}
							exam={exams.filter(item=>item._id ===currentId)[0]}
							universityId={savedUniversityId}
							courseId={savedCourseId}
							courseCode={coursesOfCurrentUniversity.find(course => course._id === savedCourseId)?.courseCode}
						/>}
					</TabPane>))}
			</Tabs>
		</Card>
	)
}
export default connect(({ university, timetable }: { university: UniversityModelState, timetable: ITimetableModelState }) => ({
	courseListOfMyUniversity: university.course,
	timetables: timetable.timetables
}))(TimetableDetail);
export { TimetableDetail };

import CourseIcon from '@/assets/omni2.0-course-icon.svg';
import UniIcon from '@/assets/omni2.0-uni-icon.svg';
import UniversityModal from '@/components/UniversityModal/UniversityModal';
import {
	UNIVERSITY_LIST
} from '@/constants/storeLocation';
import {
	TIMETABLE_TYPE, COURSE_MESSAGE
} from '@/constants/timetable';
import {
	IUniversityCourseBase
} from '@/interfaces/course';
import {
	ITimetableModelState, ITimetableState
} from '@/interfaces/timetable';
import {
	IUniversityBase, UniversityModelState
} from '@/interfaces/university';
import {
	get as storeGet,
	getSessionStorage,
	has as storeHas,
	removeSessionStorage,
	createSessionStorage
} from '@/utils/storageHelper';
import {
	compareOption
} from '@/utils/university';
import {
	Breadcrumb, Button, Card, Tabs, Typography
} from 'antd';
import {
	default as React, FC, useEffect, useState
} from 'react';
import {
	connect, ConnectProps, Link, useIntl
} from 'umi';
import Lecture from './components/Lecture';
import EmptyArrangement from './components/EmptyArrangement';
import ExamForm from './components/ExamForm';
import ExamList from './components/ExamList';
import HomeworkForm from './components/HomeworkForm';
import HomeworkList from './components/HomeworkList';
import CourseList from './components/CourseList';
import CourseModal from './components/Modals/CourseModal';
import styles from './TimetableDetail.less';

export interface ICreateNew {
	homework: boolean,
	course: boolean,
	exam: boolean,
}
interface TimetableDetailProps extends ConnectProps {
	courseListOfMyUniversity: Array<IUniversityCourseBase>,
	timetables: Array<ITimetableState>
}

const TimetableDetail: FC<TimetableDetailProps> = ({ dispatch, courseListOfMyUniversity, timetables }) => {
	console.log(courseListOfMyUniversity);
	console.log(timetables);
	const { Text } = Typography;
	const { TabPane } = Tabs;
	const intl = useIntl();
	const { formatMessage } = intl;
	const [showCreateClassTimetable, setShowCreateClassTimetable] = useState(false);
	const [isUniversityModalVisible, setIsUniversityModalVisible] = useState(false);
	const [universities, setUniversities] = useState([]);
	const [savedUniversityId, setSavedUniversityId] = useState(getSessionStorage('selectedUniversity')?._id);
	const [isClickedSaveUniversity, setIsClickedSaveUniversity] = useState(false);
	const [showUniversity, setShowUniversity] = useState('');
	const [showCourse, setShowCourse] = useState('');
	const [isChineseName, setIsChineseName] = useState(true);
	const [coursesOfCurrentUniversity, setCoursesOfCurrentUniversity] = useState([]);
	const [savedCourseId, setSavedCourseId] = useState(getSessionStorage('selectedCourseOfCurrentUniversity')?._id);
	const [isClickedSaveCourse, setIsClickedSaveCourse] = useState(false);
	const [isCourseModalVisible, setIsCourseModalVisible] = useState(false);
	const [currentId, setCurrentId] = useState();
	const [isFromCourseTimetable, setIsFromCourseTimetable] = useState(true);
	const [isEmpty, setIsEmpty] = useState({
		homework: true,
		course: true,
		exam: true
	});
	const [homework, setHomework] = useState<Array<ITimetableState>>([]);
	const [exams, setExams] = useState<Array<ITimetableState>>([]);
	const [courses, setCourses] = useState<Array<ITimetableState>>([]);
	const handleCreateClassTimetableOnClick = () => {
		setShowCreateClassTimetable(true);
	};
	const [createNew, setCreateNew] = useState<ICreateNew>({
		homework: false,
		course: false,
		exam: false
	});
	const [editing, setEditing] = useState<ICreateNew>({
		homework: false,
		course: false,
		exam: false
	});
	const panes = [
		{
			title: formatMessage({
				id: 'timetable.courses-total',
				defaultMessage: 'Course Arrangement'
			}),
			buttonTitle: formatMessage({
				id: 'timetable.detail.button.add.course',
				defaultMessage: 'Add Course'
			}),
			handleButtonOnClick: handleCreateClassTimetableOnClick,
			key: 'course',
			form: Lecture
		},
		{
			title: formatMessage({
				id: 'timetable.assignments-total',
				defaultMessage: 'Homework Arrangement'
			}),
			buttonTitle: formatMessage({
				id: 'timetable.detail.button.add.homework',
				defaultMessage: 'Add Homework'
			}),
			key: 'homework',
			form: HomeworkForm
		},
		{
			title: formatMessage({
				id: 'timetable.exams-total',
				defaultMessage: 'Exam Arrangement'
			}),
			buttonTitle: formatMessage({
				id: 'timetable.detail.button.add.exam',
				defaultMessage: 'Add Exam'
			}),
			key: 'exam',
			form: ExamForm
		}
	];
	useEffect(() => {
		document.getElementsByClassName('ant-tabs-tab-active')[0].parentElement.classList.add('tabs-wrapper');
		return () => {
			removeSessionStorage('selectedCourseOfCurrentUniversity');
			removeSessionStorage('selectedUniversity');
		};
	}, []);

	useEffect(() => {
		if (isFromCourseTimetable) {
			setSavedUniversityId(getSessionStorage('selectedUniversity')?._id);
			setSavedCourseId(getSessionStorage('selectedCourseOfCurrentUniversity')?._id);
		}
	}, [savedUniversityId, savedCourseId]);

	useEffect(() => {
		const universityList: Array<IUniversityBase> = storeHas(UNIVERSITY_LIST) ? storeGet(UNIVERSITY_LIST) : [];
		setUniversities(universityList.sort(compareOption.chineseCompareAscending));
	}, []);

	useEffect(() => {
		setShowCourse((courseListOfMyUniversity[0]?.courseCode || COURSE_MESSAGE.NO_COURSE) + ' ' + (courseListOfMyUniversity[0]?.courseTitle || COURSE_MESSAGE.ADD_COURSE));
		setSavedCourseId(courseListOfMyUniversity[0]?._id)
		setCoursesOfCurrentUniversity(courseListOfMyUniversity);
		setIsClickedSaveCourse(true);
	}, [courseListOfMyUniversity])

	useEffect(()=>{
		dispatch({
			type: 'timetable/fetchTimetables'
		});
		if (savedUniversityId) {
			dispatch({
				type: 'university/fetchCourseList',
				payload: savedUniversityId
			});
		}
	}, [])

	useEffect(()=>{
		const uniId = savedUniversityId;
		const courseId = savedCourseId;
		const myArrangements = timetables.filter(timetable =>
			uniId === timetable.university && timetable.course === courseId
		);

		const filteredHomework = myArrangements.filter(timetable => timetable.type === TIMETABLE_TYPE.ASSIGNMENT
		);
		setHomework(filteredHomework);
		filteredHomework.length ? setIsEmpty((prev)=> ({
			...prev, homework: false
		})) : setIsEmpty((prev)=> ({
			...prev, homework: true
		}));

		const filteredExams = myArrangements.filter(timetable => timetable.type === TIMETABLE_TYPE.EXAM);
		setExams(filteredExams);
		filteredExams.length ? setIsEmpty((prev)=> ({
			...prev, exam: false
		})) : setIsEmpty((prev)=> ({
			...prev, exam: true
		}));

		const filteredCourses = myArrangements.filter(timetable => timetable.type === TIMETABLE_TYPE.LECTURE);
		setCourses(filteredCourses);
		filteredCourses.length ? setIsEmpty((prev) => ({
			...prev, course: false
		})) : setIsEmpty((prev) => ({
			...prev, course: true
		}))

		setIsClickedSaveCourse(false);
		setIsClickedSaveUniversity(false);
	}, [timetables, isClickedSaveCourse, isClickedSaveUniversity])

	// change current university
	const chooseUniversity = (value: string) => {
		setIsFromCourseTimetable(false);
		setSavedUniversityId(value);
	}
	const showChangeUniversityModal = () => setIsUniversityModalVisible(true);
	const handleCancelUniversityModel = () => setIsUniversityModalVisible(false);
	const handleSaveUniversity = () => {
		setIsClickedSaveUniversity(true);
		setIsUniversityModalVisible(false);
		dispatch({
			type: 'university/fetchCourseList',
			payload: savedUniversityId
		});
		const showUniversityContent = universities.filter(university => university._id === savedUniversityId)[0];
		createSessionStorage('selectedUniversity', showUniversityContent);
		setShowUniversity(isChineseName ?
			universities.filter(university => university._id === savedUniversityId)[0]?.chineseName || 'no value'
			: universities.filter(university => university._id === savedUniversityId)[0]?.name || 'no value');
	};

	// change current course of the university
	const chooseCourse = (value: string) => {
		setIsFromCourseTimetable(false);
		setSavedCourseId(value);
	}
	const showModifyUniversityModal = () => setIsCourseModalVisible(true);
	const handleCancelCourseModel = () => setIsCourseModalVisible(false);
	const handleModifyCourse = () => {
		setIsClickedSaveCourse(true);
		setIsCourseModalVisible(false);
		if (!savedCourseId) return;
		const showCourseContent = coursesOfCurrentUniversity.filter(course => course._id === savedCourseId)[0];
		createSessionStorage('selectedCourseOfCurrentUniversity', showCourseContent);
		setShowCourse(showCourseContent?.courseCode + ' ' + showCourseContent?.courseTitle);
	}
	const edit = (type, id) => {
		setCurrentId(id);
		setCreateNew({
			...createNew, [type]: false
		});
		setEditing({
			...editing, [type]: true
		});
	}

	return (
		<Card
			bordered={false}>
			<UniversityModal visible={isUniversityModalVisible}
				onOk={handleSaveUniversity}
				onCancel={handleCancelUniversityModel}
				onChange={chooseUniversity}
				universities={universities}
			/>
			<CourseModal visible={isCourseModalVisible}
				onOk={handleModifyCourse}
				onCancel={handleCancelCourseModel}
				onChange={chooseCourse}
				courses={coursesOfCurrentUniversity}
				showUniversity={showUniversity}
				isChineseName={isChineseName}
			/>
			<Breadcrumb
				className={styles.standardNav}
				separator=">" >
				<Breadcrumb.Item><Link to='/timetable'>{formatMessage({
					id: 'menu.timetable'
				})}</Link></Breadcrumb.Item>
				<Breadcrumb.Item>
					<Text>{
						formatMessage({
							id: 'timetable.course-details'
						})}
					</Text>
				</Breadcrumb.Item>
			</Breadcrumb>
			<Breadcrumb
				className={styles.standardHeader}
				separator='|'>
				<Breadcrumb.Item>{
					formatMessage({
						id: 'menu.application.training.course.information'
					})
				}
				</Breadcrumb.Item>
				<Breadcrumb.Item>
					<img src={UniIcon}></img>
					<Text>
						{showUniversity || (getSessionStorage('selectedUniversity') ? (isChineseName ?
							`${getSessionStorage('selectedUniversity').chineseName}` :
							`${getSessionStorage('selectedUniversity').name}`) : formatMessage({
							id: 'timetable.detail.select.university',
							defaultMessage: 'No Selected University'
						}))}
					</Text>
					<Button onClick={showChangeUniversityModal}>{
						formatMessage({
							id: 'timetable.detail.button.change',
							defaultMessage: 'Change'
						})}</Button>
				</Breadcrumb.Item>
				<Breadcrumb.Item>
					<img src={CourseIcon}></img>
					<Text>
						{(getSessionStorage('selectedCourseOfCurrentUniversity') ?
							`${getSessionStorage('selectedCourseOfCurrentUniversity')?.courseCode} ${getSessionStorage('selectedCourseOfCurrentUniversity')?.courseTitle}` : showCourse)}
					</Text>
					<Button onClick={showModifyUniversityModal}>{
						formatMessage({
							id: 'timetable.detail.button.modify',
							defaultMessage: 'Modify'
						})}</Button>
				</Breadcrumb.Item>
			</Breadcrumb>
			<Tabs onChange={() => (setCreateNew({
				homework: false,
				course: false,
				exam: false
			}))}>
				{panes.map(pane => (
					<TabPane
						tab={
							<>
								{pane.title}
								<Button
									className={styles['pointer-events']}
									onClick={() => {
										setCreateNew({
											...createNew, [pane.key]: true
										});
										setIsEmpty({
											...isEmpty, [pane.key]: false
										});
										setEditing({
											...editing, [pane.key]: false
										})
										setCurrentId(null);
									} }
								>
									{pane.buttonTitle}
								</Button>
							</>
						}
						key={pane.key}>

						{
							isEmpty[pane.key] && !createNew[pane.key] &&
							<EmptyArrangement
								title={pane.title}
								create={() => {
									setCreateNew({
										...createNew, [pane.key]: true
									});
									setIsEmpty(()=>({
										...isEmpty, [pane.key]: false
									}));
								}}
							/>
						}
						{pane.key === TIMETABLE_TYPE.EXAM && exams && exams
							.map(item =>
								<ExamList key={item._id} item={item}
									edit={()=>{
										setCurrentId(item._id);
										setCreateNew((prev)=>({
											...prev, [pane.key]: true
										}))
									}}
									deleteItem={()=>{
										dispatch({
											type: 'timetable/deleteTimetableById',
											payload: item._id
										})
									}} />
							)}
						{pane.key === TIMETABLE_TYPE.HOMEWORK && homework && homework
							.map(item =>
								<HomeworkList homework={item} key={item._id.toString()} edit={()=>edit(pane.key, item._id)} deleteItem={()=>{
									dispatch({
										type: 'timetable/deleteHomeworkById',
										payload: item._id
									})
								}} />)}
						{pane.key === TIMETABLE_TYPE.COURSE && courses && courses
							.map(item =>
								<CourseList courses={item}
									key={item._id.toString()}
									edit={() => {
										setCurrentId(item._id);
										setCreateNew((prev) => ({
											...prev, [pane.key]: true
										}))
									}}
									deleteItem={()=>{
										dispatch({
											type: 'timetable/deleteTimetableById',
											payload: item._id
										})
									}}
								/>
							)
						}
						{pane.key === TIMETABLE_TYPE.HOMEWORK && editing.homework && !createNew.homework &&
							<HomeworkForm
								dispatch={dispatch}
								isEditing={editing.homework}
								setEditing = {setEditing}
								cancel={() => {
									setEditing({
										...editing, [pane.key]: false
									});
									(homework.length === 0) && setIsEmpty({
										...isEmpty, [pane.key]: true
									});
								}}
								homework={homework.filter(item => item._id === currentId)[0]}
							/>
						}
						{pane.key === TIMETABLE_TYPE.HOMEWORK && createNew.homework && !editing.homework &&
							<HomeworkForm
								dispatch={dispatch}
								setCreateNew={setCreateNew}
								cancel={() => {
									setCreateNew({
										...createNew, [pane.key]: false
									});
								}}
								universityId={savedUniversityId}
								courseId={savedCourseId}
								courseCode={coursesOfCurrentUniversity.find(course => course._id === savedCourseId)?.courseCode}
							/>}
						{pane.key === TIMETABLE_TYPE.COURSE && createNew.course && <Lecture
							key={currentId}
							dispatch={dispatch}
							cancel={() => {
								setCreateNew({
									...createNew, [pane.key]: false
								});
								if(courses.length === 0) {
									setIsEmpty({
										...isEmpty, [pane.key]: true
									});
								}
							}}
							course={courses.filter(item=>item._id ===currentId)[0]}
							universityId={savedUniversityId}
							courseId={savedCourseId}
							courseCode={coursesOfCurrentUniversity.find(course => course._id === savedCourseId)?.courseCode}
						/>}
						{pane.key === TIMETABLE_TYPE.EXAM && createNew.exam &&
						<ExamForm
							key={currentId}
							dispatch={dispatch}
							afterCancel={() => {
								setCreateNew({ ...createNew, [pane.key]: false });
								if(exams.length === 0) {
									setIsEmpty({ ...isEmpty, [pane.key]: true });
								}
							}}
							afterSave={() => {
								setCreateNew({...createNew, [pane.key]: false})
								setIsEmpty({...isEmpty, [pane.key]: false});
							}}
							exam={exams.filter(item=>item._id ===currentId)[0]}
							universityId={savedUniversityId}
							courseId={savedCourseId}
							courseCode={coursesOfCurrentUniversity.find(course => course._id === savedCourseId)?.courseCode}
						/>}
					</TabPane>))}
			</Tabs>
		</Card>
	)
}
export default connect(({ university, timetable }: { university: UniversityModelState, timetable: ITimetableModelState }) => ({
	courseListOfMyUniversity: university.course,
	timetables: timetable.timetables
}))(TimetableDetail);
export { TimetableDetail };


import {
	timeZoneCityName, getTimeWithTimezone
} from '@/utils/time';

import * as dayjs from 'dayjs';
import * as LeapYear from 'dayjs/plugin/isLeapYear';
import 'dayjs/locale/zh-cn'; // import locale
import utc from 'dayjs/plugin/utc';
import timezone from 'dayjs/plugin/timezone';

dayjs.extend(utc);
dayjs.extend(timezone);

/**
 * @description
 * format time
 */

export const formatTime = (time, template = 'DD/MM/YYYY') => {
	return dayjs(time)
		.locale('zh-cn')
		.format(template);
};

/**
 * @description
 * format time yyyy/mm/dd xx:xx:xx am/pm
 * Output empty string when no data
 */

export const formatTimeYearToSecond = time => {
	return time ? dayjs(time).format('YYYY/MM/DD hh:mm:ssa') : '';
};

/**
 * @description
 * format time
 */

export const isLeapYear = time => {
	dayjs.extend(LeapYear);
	return dayjs(time).isLeapYear();
};

/**
 * @description
 * guess the country/city of the timezone
 */

const timeZone = dayjs.tz.guess();
export const timeZoneCityName = timeZone?.split('/')[1] || 'Sydney';

/**
 * @description
 * format a range of time
 */
export const getTimeWithTimezone = (startTime, endTime, format = 'HH:mm') => {
	return `${dayjs(startTime)
		.tz(timeZone)
		.format(format)} - ${dayjs(endTime)
		.tz(timeZone)
		.format(format)}`;
};


const createTimeSlot = (day, startTime, endTime) => {
		const daySet = new Set();
		selectedTimeSlots.map((slot) => daySet.add(slot.day));
		let updatedTimeSlot = [];
		if (!daySet.has(day)) {
			updatedTimeSlot = [...selectedTimeSlots, {
				day, startTime, endTime
			}]
		} else {
			const addedTimeSlot = [{
				day, startTime, endTime
			}];
			updatedTimeSlot = selectedTimeSlots.map(timeSlot => addedTimeSlot.find(addedSlot => addedSlot.day === timeSlot.day) || timeSlot);
		}
		setSelectedTimeSlots(updatedTimeSlot);
	};

const createTimeSlot = (day, startTime, endTime) => {
		setSelectedTimeSlots([...selectedTimeSlots, {
			day,
			startTime,
			endTime
		}]);
	};